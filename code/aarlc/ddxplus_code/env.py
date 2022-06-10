import random
import copy
from tqdm import tqdm

import numpy as np

from scipy.stats import entropy

from sim_utils import (
    encode_age,
    encode_ethnicity,
    encode_geo,
    encode_race,
    encode_sex,
    load_and_check_data,
    load_csv,
)

# none value - when an inquiry for a symptom is not yet done
NONE_VAL = 0
# presence value - indicating a symptom is present
PRES_VAL = 1
# absence value - indicating a symptom is absent
ABS_VAL = -1    


class environment(object):
    """Class for simulating patients interaction for an RL project.

    The simulator is based on a csv file exported from a Synthea simulation.

    """

    def __init__(
        self,
        args,
        patient_filepath,
        train=None,
    ):
        """Init method of the simulator.

        Parameters
        ----------
        args: namespace
            arguments of the main program.
        patient_filepath: str
            path to the csv file containing generated patients
            from Synthea simulator.
        train: boolean
            if the env is used in train mode or not.
        """
        self.args = args
        self.filepath = patient_filepath
        self.use_initial_symptom_flag = not args.no_initial_evidence
        self.max_turns = args.MAXSTEP
        self.action_type = 0
        self.include_turns_in_state = args.include_turns_in_state
        self.include_race_in_state = False
        self.include_ethnicity_in_state = False
        self.use_differential_diagnosis = not args.no_differential
        self.train = args.train if train is None else train
        if not self.max_turns and self.include_turns_in_state:
            raise ValueError(
                "'max_turns' could not be None/0 if 'include_turns_in_state' is True."
            )

        # load patient data
        [
            rb,
            self.unique_symptoms,
            self.unique_pathos,
            patho_symptoms,
            self.unique_races,
            self.unique_ethnics,
            self.unique_genders,
            self.symptoms_with_multiple_answers,
            self.max_differential_len,
            self.unique_differential_pathos,
            self.unique_init_symptoms,
        ] = load_csv(self.filepath)

        if len(rb) == 0:
            raise ValueError("The provided file contains no valid patients.")

        # convert the patho_symptoms into string format
        patho_symptoms = {
            self.unique_pathos[a]: set(
                [self.unique_symptoms[b] for b in patho_symptoms[a]]
            )
            for a in patho_symptoms.keys()
        }

        # load and check symptoms and pathology
        self._load_and_check_symptoms_with_pathos(
            args.evi_meta_path,
            self.unique_symptoms,
            self.symptoms_with_multiple_answers,
            args.patho_meta_path,
            self.unique_pathos,
            patho_symptoms,
        )

        # number of demographics features (age, sex, race, ethnic)
        # the first 4 entries corespond respectively to:
        #    Age[0-7], Sex[0-1], Race[0-4], Ethnicity[0-1]
        self.num_demo_features = 10
        low_demo_values = [0] * self.num_demo_features
        high_demo_values = [1] * self.num_demo_features
        self.num_age_values = 8
        self.num_sex_values = 2
        self.num_race_values = 0
        self.num_ethnic_values = 0
        if self.include_race_in_state:
            self.num_demo_features += 5
            low_demo_values += [0] * 5
            high_demo_values += [1] * 5
            self.num_race_values = 5
        if self.include_ethnicity_in_state:
            self.num_demo_features += 2
            low_demo_values += [0] * 2
            high_demo_values += [1] * 2
            self.num_ethnic_values = 2
        if self.include_turns_in_state:
            low_demo_values = [0] + low_demo_values
            high_demo_values = [1] + high_demo_values
            self.num_demo_features += 1

        # define the action and observation spaces
        self.num_symptoms = len(self.symptom_index_2_key)
        self.num_pathos = len(self.pathology_index_2_key)
        if not (self.num_symptoms > 0 and self.num_pathos > 0):
            raise ValueError(
                "Either the number of symptoms or the number of pathologies is null."
            )
        self._define_action_and_observation_spaces(
            self.num_symptoms,
            self.num_pathos,
            self.num_demo_features,
            low_demo_values,
            high_demo_values,
            np.float32,
        )

        # turns in the interaction process (dialog)
        self.turns = 0

        self.context_size = (
            self.num_age_values
            + self.num_sex_values
            + self.num_race_values
            + self.num_ethnic_values
            + (1 if self.include_turns_in_state else 0)
        )
        self.sample_size = len(rb)
        self.idx = 0
        self.indexes = np.arange(self.sample_size)
        self.diag_size = self.num_pathos
        self.symptom_size = self.num_symptoms
        self.state_size = self.num_features - len(high_demo_values) + self.context_size
        self.cached_patients = {}
        self.cost = 1 * np.ones(self.symptom_size)
        self.earn = 1 * np.ones(self.symptom_size)
        self.action_mask = np.zeros((self.num_symptoms, self.state_size))
        self.symptom_mask = np.zeros((1, self.num_symptoms))
        self.atcd_mask = np.zeros((1, self.num_symptoms))
        for symptom_index in range(self.num_symptoms):
            start_idx = self.symptom_to_obs_mapping[symptom_index][0]
            end_idx = self.symptom_to_obs_mapping[symptom_index][1]
            self.action_mask[symptom_index, start_idx:end_idx] = 1
            is_atcd = self.symptom_data[self.symptom_index_2_key[symptom_index]].get("is_antecedent", False)
            if is_atcd:
                self.atcd_mask[0, symptom_index] = 1
            else:
                self.symptom_mask[0, symptom_index] = 1
        self.severity_mask = np.zeros((self.num_pathos,))
        severity_threshold = 3
        for patho_index in range(self.num_pathos):
            if self.pathology_data[self.pathology_index_2_key[patho_index]].get("urgence", severity_threshold) < severity_threshold:
                self.severity_mask[patho_index] = 1
        self._put_patients_data_in_cache(rb)


    def reset(self):
        self.idx = 0
        self.turns = 0
        if self.train:
            np.random.shuffle(self.indexes)
        

    def _convert_to_aarlc_format(self, rb_data):
        age = rb_data["AGE_BEGIN"]
        race = self.unique_races[rb_data["RACE"]]
        sex = self.unique_genders[rb_data["GENDER"]]
        ethnic = self.unique_ethnics[rb_data["ETHNICITY"]]
        pathology = self.unique_pathos[rb_data["PATHOLOGY"]]
        pathology_index = self.pathology_name_2_index[pathology]
        pathology_severity = self.pathology_severity_data[pathology_index]
        symptoms = [self.unique_symptoms[a] for a in rb_data["SYMPTOMS"]]
        initial_symptom = (
            None
            if len(self.unique_init_symptoms) == 0
            else self.unique_init_symptoms[rb_data["INITIAL_SYMPTOM"]]
        )
        differential_data = (
            None
            if (self.max_differential_len == -1 or not self.use_differential_diagnosis)
            else {
                self.pathology_name_2_index[
                    self.unique_differential_pathos[int(diff_data[0])]
                ]: [
                    diff_data[1],
                    diff_data[2],
                ]
                for diff_data in rb_data["DIFFERNTIAL_DIAGNOSIS"]
                if self.unique_differential_pathos[int(diff_data[0])]
                in self.pathology_name_2_index
            }
        )
        out_diff = self._compute_differential_probs(differential_data)
        differential_indices, differential_probas = out_diff
        target_state = np.ones((self.state_size), dtype=self.obs_dtype) * ABS_VAL
        if self.include_turns_in_state:
            target_state[0] = 0
        target_state = self._init_demo_features(target_state, age, sex, race, ethnic)
        binary_symptoms, present_symptoms, target_state = self.parse_target_patients(
                symptoms, target_state
        )
        result = {}
        result['bin_sym'] = binary_symptoms
        result['pres_sym'] = present_symptoms
        result['age'] = age
        result['race'] = race
        result['sex'] = sex
        result['ethnic'] = ethnic
        result['pathology_index'] = pathology_index
        result['pathology_severity'] = pathology_severity
        result['initial_symptom'] = self.symptom_name_2_index[initial_symptom]
        result['tgt_state'] = target_state
        result['differential_indices'] = differential_indices
        result['differential_probas'] = differential_probas
        return result
        

    def _put_patients_data_in_cache(self, rb):
        patients = rb.apply(lambda row_data: self._convert_to_aarlc_format(row_data), axis="columns").to_list()
        self.cached_patients = {idx: patient_data for idx, patient_data in enumerate(patients)}


    def parse_target_patients(self, symptomPat, target_state):
        binary_symptoms = []
        present_symptoms = []
        symptoms_not_listed = set(self.all_symptom_names) - set([self.get_symptom_and_value(symptom_name)[0] for symptom_name in symptomPat])
        considered_symptoms = set(symptomPat + list(symptoms_not_listed))
        for symptom_name in considered_symptoms:
            root_sympt_name, symp_val = self.get_symptom_and_value(symptom_name)
            symptom_index = self.symptom_name_2_index[root_sympt_name]
            symptom_key = self.symptom_index_2_key[symptom_index]
            # is it part of target
            is_present = symptom_name in symptomPat
            data_type = self.symptom_data_types[symptom_index]
            # first symptom should not be an antecedent
            is_antecedent = self.symptom_data[symptom_key].get("is_antecedent", False)
            if data_type == "B" and is_present and (not is_antecedent):
                binary_symptoms.append(symptom_index)

            # use default value if not present
            if data_type != "B" and not is_present:
                symp_val = self.symptom_default_value_mapping.get(symptom_index)

            f_i = self._from_symptom_index_to_frame_index(symptom_index, symp_val)

            if is_present:
                if data_type == "B":
                    present_symptoms.append(symptom_index)
                else:
                    default_value = self.symptom_default_value_mapping[symptom_index]
                    if not (str(default_value) == str(symp_val)):
                        present_symptoms.append(symptom_index)

            if data_type == "B":
                target_state[f_i] = PRES_VAL if is_present else ABS_VAL
            elif data_type == "M":
                target_state[f_i] = PRES_VAL
            else:
                # data_type == "C"
                if not (symptom_index in self.categorical_integer_symptoms):
                    target_state[f_i] = PRES_VAL
                else:
                    val_index = self.symptom_possible_val_mapping[symptom_index][
                        symp_val
                    ]
                    # rescale to 1
                    num = len(self.symptom_possible_val_mapping[symptom_index])
                    scaled = NONE_VAL + ((PRES_VAL - NONE_VAL) * (val_index + 1) / num)
                    target_state[f_i] = scaled

        present_symptoms = list(set(present_symptoms))
        return binary_symptoms, present_symptoms, target_state


    def initialize_state(self, batch_size):
        self.batch_size = batch_size
        self.batch_index = self.indexes[self.idx : self.idx+batch_size]
        self.idx += batch_size
        self.disease = []
        self.differential_indices = []
        self.differential_probas = []
        self.disease_severity = []
        self.pos_sym = []
        self.acquired_sym = []
        
        i = 0
        init_state = np.ones((batch_size, self.state_size), dtype=self.obs_dtype) * NONE_VAL
        self.target_state = np.ones((batch_size, self.state_size), dtype=self.obs_dtype) * ABS_VAL
        self.all_state = np.zeros((batch_size, self.symptom_size))
        self.target_differential = np.zeros((batch_size, self.diag_size))
        self.inquired_symptoms = np.zeros((batch_size, self.symptom_size))

        if self.include_turns_in_state:
            # normalize number of turns
            init_state[:, 0] = 0
            self.target_state[:, 0] = 0

        for item in self.batch_index:
            age = self.cached_patients[item]['age']
            race = self.cached_patients[item]['race']
            sex = self.cached_patients[item]['sex']
            ethnic = self.cached_patients[item]['ethnic']
            initial_symptom = self.cached_patients[item]['initial_symptom']
            pathology_index = self.cached_patients[item]['pathology_index']
            pathology_severity = self.cached_patients[item]['pathology_severity']
            differential_indices = copy.deepcopy(self.cached_patients[item]['differential_indices'])
            differential_probas = copy.deepcopy(self.cached_patients[item]['differential_probas'])

            binary_symptoms = self.cached_patients[item]['bin_sym']
            present_symptoms = self.cached_patients[item]['pres_sym']
            self.target_state[i, :] = self.cached_patients[item]['tgt_state'][:]

            init_state[i, :] = self._init_demo_features(init_state[i, :], age, sex, race, ethnic)
            self.disease.append(pathology_index)
            self.disease_severity.append(pathology_severity)
            self.differential_indices.append(differential_indices)
            self.differential_probas.append(differential_probas)

            # reset binary symptoms if initial symptom is from the dataset
            binary_symptoms = (
                [initial_symptom]
                if initial_symptom is not None and self.use_initial_symptom_flag
                else binary_symptoms
            )

            # only select as first indicator binary symptoms
            assert len(binary_symptoms) > 0
            first_symptom = random.choice(binary_symptoms)

            index_first_symptom = first_symptom
            first_action = self._from_symptom_index_to_inquiry_action(index_first_symptom)
            frame_index, _ = self._from_inquiry_action_to_frame_index(first_action)
            init_state[i, frame_index] = PRES_VAL

            self.all_state[i, present_symptoms] = 1
            self.inquired_symptoms[i, index_first_symptom] = 1
            if (differential_indices is not None) and (differential_probas is not None):
                self.target_differential[i, differential_indices[differential_indices != -1]] = differential_probas[differential_indices != -1]
            else:
                self.target_differential[i, pathology_index] = 1.0
            
            i += 1

        self.disease = np.array(self.disease)
        self.disease_severity = np.array(self.disease_severity)
        if len(self.differential_indices) > 0 and self.differential_indices[0] is None:
            self.differential_indices = None
            self.differential_probas = None
        else:
            self.differential_indices = np.array(self.differential_indices)
            self.differential_probas = np.array(self.differential_probas)

        return init_state, self.disease, self.differential_indices, self.differential_probas, self.disease_severity


    def step(self, s, a_p, done, right_diagnosis, agent, ent_init, threshold, ent):

        s_ = copy.deepcopy(s)
        ent_ = copy.deepcopy(ent)
        s_[~done] = (1 - self.action_mask[a_p[~done]]) * s_[~done] + self.action_mask[a_p[~done]] * self.target_state[~done]
        self.turns += 1
        if self.include_turns_in_state:
            # normalize number of turns
            s_[~done, 0] = self.turns / self.max_turns

        a_d_, p_d_ = agent.choose_diagnosis(s_)

        ent_[~done] = entropy(p_d_[~done], axis = 1)
        ent_ratio = (ent-ent_) / ent_init

        diag = (ent_ < threshold[a_d_]) & (~done)
        right_diag = (a_d_ == np.array(self.disease)) & diag 

        reward_s = self.args.mu * self.reward_func(s[:,self.context_size:], s_[:,self.context_size:], diag, a_p) 
        reward_s[ent_ratio > 0] += (self.args.nu * ent_ratio[ent_ratio > 0])
        reward_s[diag] -= (self.args.mu * 1)
        reward_s[right_diag] += (self.args.mu * 2)
        reward_s[done] = 0

        self.inquired_symptoms[~done, a_p[~done]] = 1
        
        done += diag
        right_diagnosis += right_diag
        
        return s_, reward_s, done, right_diagnosis, diag, ent_, a_d_
    
    def reward_func(self, s, s_, diag, a_p):
        
        reward = -self.cost[a_p]
        already_inquired_action = self.inquired_symptoms[range(self.inquired_symptoms.shape[0]), a_p]
        unrepeated_action = (1 - already_inquired_action)
        reward += unrepeated_action * self.cost[a_p] * 0.7
        positive_action = self.all_state[range(self.all_state.shape[0]), a_p]
        unrepeated_positive_action = positive_action * unrepeated_action
        reward += unrepeated_positive_action * self.earn[a_p]

        return reward


    def _define_action_and_observation_spaces(
        self,
        num_symptoms,
        num_pathos,
        num_demo_features,
        low_demo_values,
        high_demo_values,
        obs_dtype,
    ):
        """ Utility function for defining the enviroment action and observation spaces.

        It define the action and observation spaces for this Gym environment.

        Parameters
        ----------
        num_symptoms: int
            number of possible symptoms.
        num_pathos: int
            number of possible pathos.
        num_demo_features: int
            number of features corresponditing to demographic data.
        low_demo_values: list
            low values for demographic features.
        high_demo_values: list
            high values for demographic features.
        obs_dtype: dtype
            dtype of the observation data.

        Returns
        -------
        None
        """

        if self._has_int_action():
            num_actions = num_symptoms + num_pathos
            self.num_actions = [num_actions]
        else:
            self.num_actions = [2, num_symptoms, num_pathos]

        msg = "the length of low/high_demo_values must match num_demo_features."
        assert len(low_demo_values) == len(high_demo_values) == num_demo_features, msg

        # low and high values of each entry of the observation/state space
        low_val = list(low_demo_values)
        high_val = list(high_demo_values)

        # low and high values of each entry of the observation/state space
        # dedicated to symptoms
        symp_low_val = []
        symp_high_val = []

        # mapping the symptom index to the [start, end] indices in the obs data
        symptom_to_obs_mapping = {}

        # mapping the symptom index to possible values
        symptom_possible_val_mapping = {}

        # mapping the symptom index to the symptom types
        symptom_data_types = {}

        # mapping the symptom index to the symptom default value
        symptom_default_value_mapping = {}

        # integer based categorical value
        categorical_integer_symptoms = set()

        for idx in range(len(self.symptom_index_2_key)):
            key = self.symptom_index_2_key[idx]
            data_type = self.symptom_data[key].get("type-donnes", "B")
            possible_values = self.symptom_data[key].get("possible-values", [])
            default_value = self.symptom_data[key].get("default_value", None)
            start_obs_idx = len(symp_low_val) + num_demo_features
            symptom_data_types[idx] = data_type
            num_elts = len(possible_values)
            if num_elts > 0:
                assert default_value in possible_values

            if data_type == "B":
                # binary symptom
                symp_low_val.append(min(NONE_VAL, PRES_VAL, ABS_VAL))
                symp_high_val.append(max(NONE_VAL, PRES_VAL, ABS_VAL))
                symptom_to_obs_mapping[idx] = [start_obs_idx, start_obs_idx + 1]
            elif data_type == "C":
                # categorical symptom
                assert num_elts > 0
                if isinstance(possible_values[0], str):
                    for k in range(num_elts):
                        symp_low_val.append(min(NONE_VAL, PRES_VAL, ABS_VAL))
                        symp_high_val.append(max(NONE_VAL, PRES_VAL, ABS_VAL))
                    symptom_to_obs_mapping[idx] = [
                        start_obs_idx,
                        start_obs_idx + num_elts,
                    ]
                else:
                    # integer value
                    categorical_integer_symptoms.add(idx)
                    symp_low_val.append(min(NONE_VAL, PRES_VAL))
                    symp_high_val.append(max(NONE_VAL, PRES_VAL))
                    symptom_to_obs_mapping[idx] = [start_obs_idx, start_obs_idx + 1]
                symptom_possible_val_mapping[idx] = {
                    a: i for i, a in enumerate(possible_values)
                }
                symptom_default_value_mapping[idx] = default_value
            elif data_type == "M":
                # multi-choice symptom
                assert num_elts > 0
                for k in range(num_elts):
                    symp_low_val.append(min(NONE_VAL, PRES_VAL, ABS_VAL))
                    symp_high_val.append(max(NONE_VAL, PRES_VAL, ABS_VAL))
                symptom_to_obs_mapping[idx] = [start_obs_idx, start_obs_idx + num_elts]
                symptom_possible_val_mapping[idx] = {
                    a: i for i, a in enumerate(possible_values)
                }
                symptom_default_value_mapping[idx] = default_value
            else:
                raise ValueError(
                    f"Symptom key: {key} - Unknown data type: {data_type}."
                )

        low_val.extend(symp_low_val)
        high_val.extend(symp_high_val)
        self.obs_dtype = obs_dtype
        self.num_features = len(low_val)
        self.symptom_to_obs_mapping = symptom_to_obs_mapping
        self.symptom_possible_val_mapping = symptom_possible_val_mapping
        self.symptom_data_types = symptom_data_types
        self.symptom_default_value_mapping = symptom_default_value_mapping
        self.categorical_integer_symptoms = categorical_integer_symptoms

    def get_symptom_to_observation_mapping(self):
        """Utility to get the index range in the state space associated to each symptom.

        Utility function to get the mapping from all symptom indices to index ranges
        associated to those symptoms in the observation space.

        Parameters
        ----------

        Returns
        -------
        result: dict
            the mapping from symptom index to index range associated to
            that symptom in the observation space.

        """
        return self.symptom_to_obs_mapping

    def get_pathology_severity(self):
        """Utility to get the severity associated to each pathology.

        Parameters
        ----------

        Returns
        -------
        result: list
            the severity associated to each pathology. The pathologies are
            identified by their index in the list.

        """
        return self.pathology_severity_data

    def get_evidence_default_value_in_obs(self):
        """Utility to get the evidence default value in observation frame.

        Parameters
        ----------

        Returns
        -------
        result: list of tuple (pos, value) where the ith entru correspond
            to the position and the value in the observation frame informing
            that the ith symptom is missing.

        """
        return self.symptom_defaul_in_obs

    def _has_int_action(self):
        """Utility to check if the actions are of type int or tuple of int.

        Parameters
        ----------

        Returns
        -------
        result: bool
            True if actions are of type int.

        """
        return self.action_type == 0

    def _has_multi_choice_symptoms(self, symptom_data):
        """Utility to check if the simulator support multi-choice symptoms.

        It checks if the the provided symptom data contains
        symptoms associated with multi-choice answers.
        E.g.: `douleurxx_endroits` which coresspond to the body parts
        where you have pain.

        Parameters
        ----------
        symptom_data: dict
            the json dict describing the symptom data provided to the
            simulator.

        Returns
        -------
        result: bool
            True if the symptom data contains multi-choice symptoms.

        """
        for k in symptom_data.keys():
            if symptom_data[k].get("type-donnes", "B") == "M":
                return True
        return False

    def get_hierarchical_symptom_dependency(self):
        """Get the groups of symptoms that depend on some master symptoms.

        Returns
        -------
        result: dict
            dictionnary of the groups of dependent symptom indices. The key of
            this dictionary is the index of the master symptom.
        """
        return self.all_linked_symptom_indices

    def _get_linked_symptoms(self, symptom_data):
        """Get the groups of symptoms that are linked together.

        Symptoms are linked together if they share the same code_question.

        Parameters
        ----------
        symptom_data: dict
            dictionary representing the loaded symptom JSON file.

        Returns
        -------
        result: dict
            dictionnary of the groups of linked symptoms. The key of
            this dictionary is the base symptom.

        """
        result = {}
        for k in symptom_data.keys():
            code_question = symptom_data[k].get("code_question", k)
            if code_question not in result:
                result[code_question] = []
            result[code_question].append(k)

        # Eliminate entries with just one element in the list as
        # those are independent questions.
        # Also, retrieve the base question (it is the one equals to code_question)
        # and eliminate it from the related list (it already serves as key)
        all_keys = list(result.keys())
        for k in all_keys:
            if len(result[k]) == 1:
                result.pop(k)
            else:
                assert k in result[k]
                result[k].remove(k)

        # return the computed map
        return result

    def _load_and_check_symptoms_with_pathos(
        self,
        symptom_filepath,
        unique_symptoms,
        symptoms_with_multiple_answers,
        condition_filepath,
        unique_pathos,
        patho_symptoms,
    ):
        """Check symptom/condition JSON file validity against the provided patient file.

        It loads the symptom/pathology data and check
        if they are compliant with the ones defined
        in the `unique_symptoms`/`unique_pathos` provided
        in the patient file.

        Parameters
        ----------
        symptom_filepath: str
            path to a json file containing the symptom data.
        unique_symptoms: list
            a list of unique symptoms within the provided patient data.
        symptoms_with_multiple_answers: list
            a list of unique symptoms with multiple answer within the provided
            patient data.
        condition_filepath: str
            path to a json file containing the pathology data.
        unique_pathos: list
            a list of unique pathologies within the provided patient data.
        patho_symptoms: dict
            a mapping from a pathology to a set of symptoms describing that
            pathology as derived from the patient data.

        Returns
        -------
        None

        """

        # load symptoms
        symptom_infos = load_and_check_data(
            symptom_filepath, unique_symptoms, key_name="name"
        )
        self.symptom_index_2_key = symptom_infos[0]
        self.symptom_name_2_index = symptom_infos[1]
        self.symptom_data = symptom_infos[2]
        self.multi_choice_flag = self._has_multi_choice_symptoms(self.symptom_data)
        self.all_symptom_names = [
            self.symptom_data[k]["name"] for k in self.symptom_data.keys()
        ]
        self.all_linked_symptoms = self._get_linked_symptoms(self.symptom_data)
        # transform 'self.all_linked_symptoms' from str into symptom indices
        self.all_linked_symptom_indices = {
            self.symptom_name_2_index[self.symptom_data[base_symp_key]["name"]]: [
                self.symptom_name_2_index[self.symptom_data[linked_symp_key]["name"]]
                for linked_symp_key in self.all_linked_symptoms[base_symp_key]
            ]
            for base_symp_key in self.all_linked_symptoms.keys()
        }
        # reverse the linked symptoms map: from linked_symptom => base_symptom
        self.all_linked_reverse_symptom_indices = {
            linked_symp_idx: base_symp_idx
            for base_symp_idx in self.all_linked_symptom_indices
            for linked_symp_idx in self.all_linked_symptom_indices[base_symp_idx]
        }
        for a in symptoms_with_multiple_answers:
            idx = self.symptom_name_2_index[a]
            key = self.symptom_index_2_key[idx]
            data_type = self.symptom_data[key].get("type-donnes", "B")
            if data_type != "M":
                raise ValueError(
                    f"Unconsistency with Symptom {a}: Occured multiple times while"
                    f" not a multiple choice symptom. {symptoms_with_multiple_answers}"
                )

        # load pathologies
        pathology_infos = load_and_check_data(
            condition_filepath, unique_pathos, key_name="condition_name"
        )
        self.pathology_index_2_key = pathology_infos[0]
        self.pathology_name_2_index = pathology_infos[1]
        self.pathology_data = pathology_infos[2]
        self.pathology_defined_symptoms = {}
        # get all pathology severity - # default severity to 0
        self.pathology_severity_data = [
            self.pathology_data[self.pathology_index_2_key[idx]].get("urgence", 0)
            for idx in range(len(self.pathology_index_2_key))
        ]

        # check if the provided df respect the symptom/patho relationships
        for key in self.pathology_data.keys():
            defined_symptom_keys = list(self.pathology_data[key]["symptoms"].keys())
            if "antecedents" in self.pathology_data[key]:
                defined_symptom_keys += list(
                    self.pathology_data[key]["antecedents"].keys()
                )
            defined_symptoms = []
            for k in defined_symptom_keys:
                symp_name = self.symptom_data[k]["name"]
                symp_type = self.symptom_data[k].get("type-donnes", "B")
                # binary symptoms
                if symp_type == "B":
                    defined_symptoms.append(symp_name)
                else:
                    # categorical or multi-choice
                    possible_values = self.symptom_data[k].get("possible-values", [])
                    for v in possible_values:
                        val_name = symp_name + "_@_" + str(v)
                        defined_symptoms.append(val_name)
            defined_symptoms = set(defined_symptoms)
            self.pathology_defined_symptoms[key] = defined_symptoms
            patho = self.pathology_data[key]["condition_name"]

            if patho in patho_symptoms:
                data_symptoms = patho_symptoms[patho]
                diff = data_symptoms - defined_symptoms
                if len(diff) > 0:
                    raise ValueError(
                        f"Unconsistency with patho {patho}: Unauthorized symptoms {diff}"
                    )

    def get_symptom_and_value(self, symptom_name):
        """Utility function to get the symptom and the associated value from csv data.

        Given a symptom, find its root and assocaited value
        for example, `douleurxx_carac_@_penible` will return
        `douleurxx_carac` and `penible`.
        Similarly, `fievre` will return `fievre` and  None (which
        is the default value for boolean symptom).

        Parameters
        ----------
        symptom_name: str
            the symptom (from csv) for which we want to retrieve the info data.

        Returns
        -------
        symptom: str
            the symptom name as defined in the config file.
        val: object
            the value associated to the symptom.

        """
        idx = symptom_name.find("_@_")
        if idx == -1:
            # boolean symptom
            return symptom_name, None
        else:
            elem_base = symptom_name[:idx]
            elem_val = symptom_name[idx + 3 :]
            base_idx = self.symptom_name_2_index.get(elem_base, -1)

            assert base_idx != -1, (
                f"The symptom {elem_base} is not defined "
                f"while receiving {symptom_name}!"
            )

            assert self.symptom_possible_val_mapping.get(base_idx)
            if not (elem_val in self.symptom_possible_val_mapping[base_idx]):
                # convert to the right type
                elem_val = int(elem_val)
                assert elem_val in self.symptom_possible_val_mapping[base_idx]

            return elem_base, elem_val

    def _init_demo_features(self, frame, age, sex, race, ethnic):
        """Set the demographic features in the provided observation frame.

        Parameters
        ----------
        frame: np.array
            the observation frame to be updated.
        age: int
            the age of the patient.
        sex: str
            the sex of the patient.
        race: str
            the race of the patient.
        ethnic: str
            the ethnic of the patient.

        Returns
        -------
        result: np.array
            the updated observation frame.

        """
        tmp_init_idx = 1 if self.include_turns_in_state else 0

        # set the age
        frame[tmp_init_idx + encode_age(age)] = 1
        # set the sex
        frame[tmp_init_idx + self.num_age_values + encode_sex(sex)] = 1
        # set the race
        if self.include_race_in_state:
            val = self.num_age_values + self.num_sex_values
            frame[tmp_init_idx + val + encode_race(race)] = 1
        # set the ethnicity
        if self.include_ethnicity_in_state:
            val = self.num_age_values + self.num_sex_values + self.num_race_values
            frame[tmp_init_idx + val + encode_ethnicity(ethnic)] = 1
        # return the updated frame
        return frame

    def _compute_differential_probs(self, differential):
        """Compute the differential probability from the diffential scores.

        Parameters
        ----------
        differential: dict
            Map of the pathology id in the differential to its sommeOR and score
            as returned by DXA.

        Returns
        -------
        indices: np.ndarray
            the array correspondind to the pathology indices involved in the
            differential (-1 is used for padding).
        probability: np.ndarray
            the array representing the computed probabilities associated to the
            pathologies represented by `indices`.

        """
        if differential is None or not self.use_differential_diagnosis:
            return None, None
        else:
            assert len(differential) <= self.max_differential_len
            indices = np.ones(self.max_differential_len, dtype=int) * -1
            probability = np.zeros(self.max_differential_len, dtype=np.float32)
            sumProba = 0
            for i, k in enumerate(differential.keys()):
                indices[i] = k
                sommeOR, _ = differential[k]
                proba = sommeOR / (1.0 + sommeOR)
                sumProba += proba
                probability[i] = proba
            if sumProba != 0:
                probability = probability / sumProba
            else:
                probability[0 : len(differential)] = 1.0 / len(differential)

        # sort in desceding order according to proba
        s_ind = np.argsort(probability, axis=-1)
        probability = probability[s_ind[::-1]]
        indices = indices[s_ind[::-1]]

        return indices, probability

    def _from_inquiry_action_to_symptom_index(self, action):
        """Get the index of the symptom corresponding to the inquiry action.

        Parameters
        ----------
        action: int
            the action from which we want to find the corresponding
            symptom index.

        Returns
        -------
        index: int
            index of the symptom associated with the provided action.

        """
        if self._has_int_action():
            symptom_index = action
        else:
            symptom_index = action[1]

        if isinstance(symptom_index, np.ndarray):
            symptom_index = symptom_index.item()

        return symptom_index

    def _from_symptom_index_to_inquiry_action(self, symptom_index):
        """Get the inquiry action corresponding to the provided symptom index.

        Parameters
        ----------
        symptom_index: int
            the index of the symptom from which we want to find
            the corresponding action.

        Returns
        -------
        action: int
            the inquiry action associated with the provided symptom.

        """
        if self._has_int_action():
            return symptom_index
        else:
            return [0, symptom_index, 0]

    def _from_symptom_index_to_frame_index(self, symptom_index, symptom_val=None):
        """Get the frame index corresponding to the symptom_index.

        Parameters
        ----------
        symptom_index: int
            the index of the symptom from which we want to find
            the corresponding action.
        symptom_val: obj
            the value associated to the the symptom indexed by
            the provided index. Default: None

        Returns
        -------
        index: int
            frame index associated with the provided action.

        """
        data_type = self.symptom_data_types[symptom_index]

        if data_type == "B":
            return self.symptom_to_obs_mapping[symptom_index][0]
        elif data_type == "C" and (symptom_index in self.categorical_integer_symptoms):
            return self.symptom_to_obs_mapping[symptom_index][0]
        else:
            if symptom_val is None:
                raise ValueError("The expected value is NOT supposed to be None.")
            idx = self.symptom_possible_val_mapping[symptom_index][symptom_val]
            return self.symptom_to_obs_mapping[symptom_index][0] + idx

    def _from_inquiry_action_to_frame_index(self, action):
        """Get the frame index corresponding to the inquiry action.

        Parameters
        ----------
        action: int
            the action from which we want to find the corresponding
            frame index.

        Returns
        -------
        indices: couple of int
            frame indices interval [start, end] associated with
            the provided action.

        """
        symptom_index = self._from_inquiry_action_to_symptom_index(action)

        start_idx = self.symptom_to_obs_mapping[symptom_index][0]
        end_idx = self.symptom_to_obs_mapping[symptom_index][1]

        return start_idx, end_idx

if __name__ == '__main__':
    class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super(AttrDict, self).__init__(*args, **kwargs)
            self.__dict__ = self
    
    args = AttrDict({
        'no_initial_evidence': False,
        'MAXSTEP': 30,
        'include_turns_in_state': True,
        'no_differential': False,
        'evi_meta_path': "./release_evidences.json",
        'patho_meta_path': "./release_conditions.json",
        'train': True,
    })
    patient_filepath = "./release_validate_patients.zip"
    np.random.seed(1000)
    random.seed(1000)
    env = environment(args, patient_filepath)
