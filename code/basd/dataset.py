import random
from collections import defaultdict
import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

from env import environment, NONE_VAL, PRES_VAL
from metrics import compute_metrics


class ASDPatientInteractionSimulator(environment):
    """docstring for ASDPatientInteractionSimulator"""

    def __init__(self, *args, **kwargs):
        super(ASDPatientInteractionSimulator, self).__init__(*args, **kwargs)

    def is_atcd(self, action):
        symptom_index = self._from_inquiry_action_to_symptom_index(action)
        symptom_key = self.symptom_index_2_key[symptom_index]
        return self.symptom_data[symptom_key].get("is_antecedent", False)

    def re_seed(self, seed):
        self.np_random = random.Random(seed)

    def get_symptom_index_from_action(self, action):
        symptom_index = self._from_inquiry_action_to_symptom_index(action)
        return symptom_index
        
    def get_differential_data(self, index):
        indices = self.cached_patients[index]['differential_indices']
        probas = self.cached_patients[index]['differential_probas']
        pathology_index = self.cached_patients[index]['pathology_index']
        differential = np.zeros((self.diag_size,))
        if (indices is not None) and (probas is not None):
            differential[indices[indices != -1]] = probas[indices != -1]
        else:
            differential[pathology_index] = 1.0
        return differential
        
    def get_demo_data(self, index):
        return (
            self.cached_patients[index]['age'], self.cached_patients[index]['sex'],
            self.cached_patients[index]['race'], self.cached_patients[index]['ethnic']
        )

    def get_relevant_symptoms_and_antecedent_from_patient(self, index):
        """Get the symptoms/antecedents within the provided patient data.

        Parameters
        ----------
        index: int
            index of the patient in the dataset

        Returns
        -------
        symptoms: list of int
            relevant symptoms found.
        antecedents: list of int
            relevant antecedents found.
        binary_symp: list of int
            relevant binary symptoms found.

        """
        symptoms = list(self.cached_patients[index]['pres_sym'])
        antecedents = list(self.cached_patients[index]['pres_atcd'])
        binary_symptoms = list(self.cached_patients[index]['bin_sym'])
        initial_symptom = self.cached_patients[index]['initial_symptom']
        return symptoms, antecedents, binary_symptoms, initial_symptom

    def sample(self, alist, num):
        return random.sample(alist, num)

    def set_frame_val(self, frame, index, pres_s, evidences, interaction_length):
        target_state = self.cached_patients[index]['tgt_state']
        
        # setting the presenting symptoms
        a = pres_s
        frame = (1 - self.action_mask[a]) * frame + self.action_mask[a] * target_state
        
        # setting all the other evidences
        for symptom_index in evidences:
            a = symptom_index
            frame = (1 - self.action_mask[a]) * frame + self.action_mask[a] * target_state

        # normalize number of turns
        if self.include_turns_in_state:
            frame[0] = min(len(evidences), interaction_length) / interaction_length
        return frame

    def set_user_agent_action(
        self, sampled_evidences, pos_evidences, neg_evidences, frame
    ):
        # self-report:0, confirm:1, deny:2
        # initiate: 3, request: 4
        cur_sampled_evidence = (
            None
            if len(sampled_evidences) == 0
            else self.sample(sampled_evidences, 1)[0]
        )
        user_agent_action = np.zeros((5), dtype=self.obs_dtype)
        if len(sampled_evidences) == 0:
            user_agent_action[0] = 1
            user_agent_action[3] = 1
        else:
            assert len(sampled_evidences) > 0
            if cur_sampled_evidence in pos_evidences:
                user_agent_action[1] = 1
                user_agent_action[4] = 1
            else:
                user_agent_action[2] = 1
                user_agent_action[4] = 1

        return user_agent_action

    def get_data_at_index(
        self,
        index,
        interaction_length,
        min_recovery_evidence_ratio=1.0,
        only_predict_differential_at_end=False,
    ):
        """Get the eventually corrupted data at the provided index.

        Parameters
        ----------
        index: int
            the index of the patient in the DB to be collected.
        interaction_length: int
            max interaction length
        min_recovery_evidence_ratio: float
            minimum ratio of recovered evidences to end up the interactions.
            Default: 1.0.
        only_predict_differential_at_end: boolean
            flag indicating whether or not to predict differential only at end.
            Default: False.

        Returns
        -------
        observation: np.ndarray
            The observation cooresponding to the patient. This corresponds
            to his/her age, sex, race, ethnic as well as a randomly
            selected symptom.

        """
        assert interaction_length > 0
        # assert self.include_turns_in_state
        assert min_recovery_evidence_ratio >= 0 and min_recovery_evidence_ratio <= 1.0

        # get positive antecedents and symtoms indices
        pos_symt, pos_atcd, bs, init_symp = self.get_relevant_symptoms_and_antecedent_from_patient(index)
        age, sex, race, ethnic = self.get_demo_data(index)
        diff = self.get_differential_data(index)
        bs = ([init_symp] if init_symp is not None and self.use_initial_symptom_flag else bs)
        
        # presenting symptom
        assert len(bs) > 0
        pres_s = self.sample(bs, 1)[0]

        curr_positive_evidence = set(pos_symt + pos_atcd)
        curr_positive_evidence.remove(pres_s)
        curr_positive_evidence_lst = list(curr_positive_evidence)

        max_evidences = min(interaction_length, len(curr_positive_evidence))
        sample_pos_num_sym = (
            0
            if max_evidences <= 0
            else self.sample(range(max_evidences), 1)[0]
        )
        sampled_pos_sym = (
            []
            if sample_pos_num_sym <= 0
            else self.sample(curr_positive_evidence_lst, sample_pos_num_sym)
        )

        sample_pos_num_ag = self.sample(range(max_evidences + 1), 1)[0]
        sampled_pos_ag = (
            []
            if sample_pos_num_ag <= 0
            else self.sample(curr_positive_evidence_lst, sample_pos_num_ag)
        )

        num_neg_sym = interaction_length - len(sampled_pos_sym)
        num_neg_ag = interaction_length - len(sampled_pos_ag)
        num_sympt = len(self.symptom_index_2_key)
        curr_negative_evidence = set(range(num_sympt)) - curr_positive_evidence
        curr_negative_evidence.remove(pres_s)
        curr_negative_evidence_lst = list(curr_negative_evidence)

        sample_neg_num_sym = (
            0
            if num_neg_sym <= 0
            else self.sample(range(num_neg_sym), 1)[0]
        )
        sampled_neg_sym = (
            []
            if sample_neg_num_sym <= 0
            else self.sample(curr_negative_evidence_lst, sample_neg_num_sym)
        )

        sample_neg_num_ag = (
            0
            if num_neg_ag <= 0
            else self.sample(range(num_neg_ag), 1)[0]
        )
        sampled_neg_ag = (
            []
            if sample_neg_num_ag <= 0
            else self.sample(curr_negative_evidence_lst, sample_neg_num_ag)
        )

        sampled_evidences_sym_inp = set(sampled_pos_sym + sampled_neg_sym)
        sampled_evidences_ag_inp = set(sampled_pos_ag + sampled_neg_ag)

        # create the frame
        frame_ag = np.ones((self.state_size,), dtype=self.obs_dtype) * NONE_VAL
        frame_sym = np.ones((self.state_size,), dtype=self.obs_dtype) * NONE_VAL

        # set the demographic features
        frame_sym = self._init_demo_features(frame_sym, age, sex, race, ethnic)
        frame_ag = self._init_demo_features(frame_ag, age, sex, race, ethnic)

        # set the values of the samples evidences in the frame
        # import pdb;pdb.set_trace()
        il = interaction_length
        frame_sym = self.set_frame_val(
            frame_sym, index, pres_s, sampled_evidences_sym_inp, il
        )
        frame_ag = self.set_frame_val(
            frame_ag, index, pres_s, sampled_evidences_ag_inp, il
        )

        # set previous agent action and user reply
        # import pdb;pdb.set_trace()
        user_agent_action_ag = self.set_user_agent_action(
            sampled_evidences_ag_inp, sampled_pos_ag, sampled_neg_ag, frame_ag
        )
        user_agent_action_sym = self.set_user_agent_action(
            sampled_evidences_sym_inp, sampled_pos_sym, sampled_neg_sym, frame_sym
        )

        # import pdb;pdb.set_trace()
        input_ag = np.concatenate((frame_ag, user_agent_action_ag))
        input_sym = np.concatenate((frame_sym, user_agent_action_sym))

        # set the target
        remain_pos = curr_positive_evidence - set(sampled_pos_sym)
        if len(remain_pos) > 0:
            sym_gt = np.array([self.sample(list(remain_pos), 1)[0]])
        else:
            assert len(curr_positive_evidence) == 0
            sym_gt = np.array([pres_s])
            start_index = self.symptom_to_obs_mapping[pres_s][0]
            input_sym[start_index] = NONE_VAL

        # conclude: 0, query: 1
        nval = len(curr_positive_evidence - set(sampled_pos_ag))
        is_min_recovered = (
            nval <= int((1 - min_recovery_evidence_ratio) * len(curr_positive_evidence))
        )
        ag_gt = np.array([
            0
            if nval == 0
            else (1 if not is_min_recovered else self.sample([0, 1], 1)[0])
        ])
        # is differential need to be used for updating network
        is_diff = np.array([1]) if not only_predict_differential_at_end else 1 - ag_gt
        # import pdb;pdb.set_trace()
        return input_ag, ag_gt, input_sym, sym_gt, is_diff, diff

    def reset_(self, index):
        """Reset an interaction between a patient and an automated agent.
        """
        # print("in sim")
        init_state = np.ones((self.state_size, ), dtype=self.obs_dtype) * NONE_VAL
        age, sex, race, ethnic = self.get_demo_data(index)
        bs = list(self.cached_patients[index]['bin_sym'])
        init_symp = self.cached_patients[index]['initial_symptom']
        bs = ([init_symp] if init_symp is not None and self.use_initial_symptom_flag else bs)
        assert len(bs) > 0
        pres_s = self.sample(bs, 1)[0]
        index_first_symptom = pres_s
        
        init_state[:] = self._init_demo_features(init_state, age, sex, race, ethnic)
        first_action = self._from_symptom_index_to_inquiry_action(index_first_symptom)
        frame_index, _ = self._from_inquiry_action_to_frame_index(first_action)
        init_state[frame_index] = PRES_VAL
        
        if self.include_turns_in_state:
            # normalize number of turns
            init_state[0] = 0
            
        self.turns = 0
        self.target_pathology_index = self.cached_patients[index]['pathology_index']
        self.target_pathology_severity = self.cached_patients[index]['pathology_severity']
        self.target_differential_indices = copy.deepcopy(self.cached_patients[index]['differential_indices'])
        self.target_differential_probas = copy.deepcopy(self.cached_patients[index]['differential_probas'])
        self.target_differential_dist = self.get_differential_data(index)
        self.target_frame_symptoms = copy.deepcopy(self.cached_patients[index]['tgt_state'])
        self.target_inquired_symptoms = np.zeros((self.symptom_size, ))
        self.target_all_symptoms = np.zeros((self.symptom_size, ))
        self.target_first_action = first_action
        
        present_evidences = self.cached_patients[index]['pres_evi']
        self.target_inquired_symptoms[index_first_symptom] = 1
        self.target_all_symptoms[present_evidences] = 1

        return init_state
        
    def step_(self, s, action):
        a = self.get_symptom_index_from_action(action)
        s_ = copy.deepcopy(s)
        s_ = (1 - self.action_mask[a]) * s_ + self.action_mask[a] * self.target_frame_symptoms
        self.turns += 1
        if self.include_turns_in_state:
            # normalize number of turns
            s_[0] = self.turns / self.max_turns

        self.target_inquired_symptoms[a] = 1
        
        return s_


class SimPaDatasetTrain(Dataset):
    """Class representing Dataset defined from environment for asd training purposes.
    """

    def __init__(
        self,
        env,
        interaction_length,
        min_recovery_evidence_ratio=1.0,
        only_predict_differential_at_end=False,
    ):
        """Init method of dataset.

        Parameters
        ----------
        env: PatientInteractionSimulator
            the environment from which the dataset is derived.
        interaction_length: int
            max interaction length
        min_recovery_evidence_ratio: float
            minimum ratio of recovered evidences to end up the interactions.
            Default: 1.0.
        only_predict_differential_at_end: boolean
            flag indicating whether or not to predict differential only at end.
            Default: False.
        """
        self.env = env
        self.interaction_length = interaction_length
        self.min_recovery_evidence_ratio = min_recovery_evidence_ratio
        self.only_predict_differential_at_end = only_predict_differential_at_end

    def __getitem__(self, idx):
        """Get the item at the provided index.

        Parameters
        ----------
        idx: int
            the index of interest.
        """
        return self.env.get_data_at_index(
            idx,
            self.interaction_length,
            self.min_recovery_evidence_ratio,
            self.only_predict_differential_at_end,
        )

    def __len__(self):
        """Get the len of the dataset.
        """
        return self.env.sample_size


class SimPaDataset:
    """Class representing Dataset defined from environment for asd evaluation purposes.
    """

    def __init__(self, env, interaction_length):
        """Init method of dataset.

        Parameters
        ----------
        env: PatientInteractionSimulator
            the environment from which the dataset is derived.
        interaction_length: int
            max interaction length
        """
        self.env = env
        self.total_length = interaction_length

    def sample(self, alist, num):
        return random.sample(alist, num)


    @torch.no_grad()
    def evaluate2(
        self,
        agent,
        total_length=0,
        num_eval_trajs=0,
        eval_patient_ids=None,
        seed=None,
        cuda_idx=None,
        thres=0.5,
        masked_inquired_actions=True,
        compute_metrics_flag=True,
        batch_size=1,
    ):
        device = torch.device(
            f"cuda:{cuda_idx}"
            if torch.cuda.is_available() and cuda_idx is not None
            else "cpu"
        )
        random.seed(seed)
        self.env.re_seed(seed)

        total_length = self.total_length if total_length == 0 else total_length

        overall_recall = 0
        overall_il = 0
        num_agent_actions = 2
        num_user_actions = 3

        if (num_eval_trajs == 0) and (eval_patient_ids is None):
            num_eval_trajs = self.env.sample_size
            total_patients = self.env.sample_size
            eval_patient_ids = list(range(total_patients))
        elif eval_patient_ids is None:
            total_patients = self.env.sample_size
            num_eval_trajs = min(num_eval_trajs, total_patients)
            eval_patient_ids = self.sample(list(range(total_patients)), num_eval_trajs)
        elif num_eval_trajs == 0:
            assert all([i < self.env.sample_size for i in eval_patient_ids])
            num_eval_trajs = len(eval_patient_ids)

        agent = agent.to(device)
        agent.train(False)

        all_metrics = {}
        for batch_idx, patient_id in tqdm(enumerate(eval_patient_ids), total=len(eval_patient_ids), desc="Evaluation: "):

            obs = self.env.reset_(patient_id)
            
            user_offset = obs.shape[0]
            agent_offset = obs.shape[0] + num_user_actions

            # self-report:0, confirm:1, deny:2
            prev_user_act = 0
            # # initiate: 0 (3), request: 1 (4)
            prev_agent_act = 0
            

            # import pdb;pdb.set_trace()
            # print(mask)
            if masked_inquired_actions:
                num_sympt = len(self.env.symptom_index_2_key)
                mask = torch.zeros(1, num_sympt).to(device)
                mask[:, [self.env.target_first_action]] = 1

            all_diags = []
            curr_turn = 0
            valid_timesteps = []
            while curr_turn < total_length:
                inputs = np.zeros(
                    (1, obs.shape[0] + num_agent_actions + num_user_actions)
                )
                inputs[0, :user_offset] = obs
                inputs[0, user_offset + prev_user_act] = 1
                inputs[0, agent_offset + prev_agent_act] = 1
                inputs = torch.from_numpy(inputs).float().to(device)

                # import pdb;pdb.set_trace()
                symp_pred, ag_pred, patho_pred = agent(inputs, inputs)

                patho_pred = (
                    None
                    if patho_pred is None
                    else F.softmax(patho_pred, dim=-1).detach().view(-1).cpu().numpy()
                )
                all_diags.append(patho_pred)

                ag_action = torch.sigmoid(ag_pred).item()
                if ag_action < thres:
                    prev_agent_act = 0
                    break

                if masked_inquired_actions:
                    symp_pred = symp_pred.masked_fill(mask.bool(), float("-inf"))

                sym_action = torch.argmax(symp_pred, dim=1).item()
                sym_index = self.env.get_symptom_index_from_action(sym_action)

                if masked_inquired_actions:
                    mask[:, [sym_action]] = 1

                prev_agent_act = 1
                prev_user_act = (
                    1 if self.env.target_all_symptoms[sym_index] == 1
                    else 2
                )

                obs = self.env.step_(obs, sym_action)
                curr_turn += 1
                valid_timesteps.append(1)

            if prev_agent_act == 1:
                inputs = np.zeros(
                    (1, obs.shape[0] + num_agent_actions + num_user_actions)
                )
                inputs[0, :user_offset] = obs
                inputs[0, user_offset + prev_user_act] = 1
                inputs[0, agent_offset + prev_agent_act] = 1
                inputs = torch.from_numpy(inputs).float().to(device)

                # import pdb;pdb.set_trace()
                _, _, patho_pred = agent(inputs, inputs)
                patho_pred = (
                    None
                    if patho_pred is None
                    else F.softmax(patho_pred, dim=-1).detach().view(-1).cpu().numpy()
                )
                all_diags.append(patho_pred)
                
            overall_il += len(all_diags)
            overall_recall += np.sum(self.env.target_inquired_symptoms * self.env.target_all_symptoms) / max(1, len(all_diags))
                
            if compute_metrics_flag:
                all_diags = np.array([all_diags])
                valid_timesteps = np.array([valid_timesteps])
                batch_metrics = compute_metrics(
                    self.env.target_differential_dist.reshape(1, -1), np.array([self.env.target_pathology_index]),
                    all_diags[:, -1], all_diags, valid_timesteps,
                    self.env.target_all_symptoms.reshape(1, -1), self.env.target_inquired_symptoms.reshape(1, -1),
                    self.env.symptom_mask, self.env.atcd_mask, self.env.severity_mask, tres=0.01
                )
                all_metrics = {a: (batch_idx / (batch_idx + 1)) * all_metrics.get(a, 0) + (1 / (batch_idx + 1)) * batch_metrics[a] for a in batch_metrics.keys()}

        results_dict = {
            "average_evidence_recall": overall_recall / max(1, len(eval_patient_ids)),
            "average_step": overall_il / max(1, len(eval_patient_ids)),
        }
        if compute_metrics_flag:
            all_metrics = {a: all_metrics[a].tolist() if hasattr(all_metrics[a], "tolist") else all_metrics[a] for a in all_metrics.keys()}
            results_dict.update(all_metrics) 

    @torch.no_grad()
    def evaluate(
        self,
        agent,
        total_length=0,
        num_eval_trajs=0,
        eval_patient_ids=None,
        seed=None,
        cuda_idx=None,
        thres=0.5,
        masked_inquired_actions=True,
        compute_metrics_flag=True,
        batch_size=1,
    ):
        device = torch.device(
            f"cuda:{cuda_idx}"
            if torch.cuda.is_available() and cuda_idx is not None
            else "cpu"
        )
        random.seed(seed)
        self.env.re_seed(seed)

        total_length = self.total_length if total_length == 0 else total_length

        overall_recall = 0
        overall_il = 0
        num_agent_actions = 2
        num_user_actions = 3

        if (num_eval_trajs == 0) and (eval_patient_ids is None):
            num_eval_trajs = self.env.sample_size
            total_patients = self.env.sample_size
            eval_patient_ids = list(range(total_patients))
        elif eval_patient_ids is None:
            total_patients = self.env.sample_size
            num_eval_trajs = min(num_eval_trajs, total_patients)
            eval_patient_ids = self.sample(list(range(total_patients)), num_eval_trajs)
        elif num_eval_trajs == 0:
            assert all([i < self.env.sample_size for i in eval_patient_ids])
            num_eval_trajs = len(eval_patient_ids)

        agent = agent.to(device)
        agent.train(False)

        all_metrics = {}
        num_batches = len(eval_patient_ids) // batch_size
        if len(eval_patient_ids) % batch_size > 0:
            num_batches += 1
        print(f"Number patients: {len(eval_patient_ids)} - Number Batches: {num_batches} - Batch Size: {batch_size} - NoRandom: {self.env.use_initial_symptom_flag}")
        for batch_idx in tqdm(range(num_batches), total=num_batches, desc="Evaluation: "):
                    
            patientIndices = eval_patient_ids[batch_idx * batch_size: min(len(eval_patient_ids), (batch_idx + 1) * batch_size)]
            
            obs, true_disease, _, _, _ = self.env.initialize_state(indices=patientIndices)
            
            # orig_tgt_state = copy.deepcopy(self.env.target_state)
            # tmp_val = "\n".join([f"{i} - {np.nonzero(self.env.action_mask[i])}" for i in range(self.env.action_mask.shape[0])])
            # print(f"Action Mask: \n {tmp_val}")
            # print("Categorical Integer Value mapping")
            # for i in self.env.categorical_integer_symptoms:
            #     print(f"{i} - {self.env.symptom_possible_val_mapping[i]}")
            # tmp_val = "\n".join([f"{i} - {a}" for i, a in enumerate(orig_tgt_state.T.tolist())])
            # print(f"Target_state: \n {tmp_val}")
                     
            user_offset = obs.shape[1]
            agent_offset = obs.shape[1] + num_user_actions

            # self-report:0, confirm:1, deny:2
            prev_user_act = np.array([0] * obs.shape[0]).astype(int)
            # # initiate: 0 (3), request: 1 (4)
            prev_agent_act = np.array([0] * obs.shape[0]).astype(int)
            
            done = np.zeros((obs.shape[0],)).astype(bool)

            # import pdb;pdb.set_trace()
            # print(mask)
            if masked_inquired_actions:
                mask = torch.from_numpy(copy.deepcopy(self.env.inquired_symptoms)).to(device)

            all_diags = []
            curr_turn = 0
            valid_timesteps = []
            final_diagnosis = np.zeros((obs.shape[0], self.env.diag_size))
            while curr_turn < total_length:
                # print(f"Turn {curr_turn} - obs: {np.nonzero(obs)} - values: {obs[np.nonzero(obs)]} - inchanged: {np.all(orig_tgt_state == self.env.target_state)}")
                valid_timesteps.append(np.zeros((obs.shape[0],)).astype(bool))
                inputs = np.zeros(
                    (obs.shape[0], obs.shape[1] + num_agent_actions + num_user_actions)
                )
                inputs[:, :user_offset] = obs
                inputs[range(obs.shape[0]), user_offset + prev_user_act] = 1
                inputs[range(obs.shape[0]), agent_offset + prev_agent_act] = 1
                inputs = torch.from_numpy(inputs).float().to(device)

                # import pdb;pdb.set_trace()
                symp_pred, ag_pred, patho_pred = agent(inputs, inputs)

                patho_pred = (
                    None
                    if patho_pred is None
                    else F.softmax(patho_pred, dim=-1).detach().view(obs.shape[0], -1).cpu().numpy()
                )
                all_diags.append(patho_pred if patho_pred is not None else (np.ones((obs.shape[0], self.env.diag_size))/self.env.diag_size))
                
                # tmp_val = "\n".join([f"{i} - {a}" for i, a in enumerate(all_diags[-1].T.tolist())])
                # print(f"Turn {curr_turn} - pathopred: \n {tmp_val}")

                ag_action = torch.sigmoid(ag_pred).detach().view(-1).cpu().numpy()
                
                # print(f"Turn {curr_turn} - ag_action: {ag_action.tolist()}")
                
                should_stop = ag_action < thres
                newly_done = np.logical_and(~done, should_stop)
                done[should_stop] = True
                prev_agent_act[done] = 0
                prev_user_act[done] = 0
                final_diagnosis[newly_done] = all_diags[-1][newly_done]
                if all(done):
                    if len(valid_timesteps) > 1:
                        valid_timesteps.pop()
                    break


                # tmp_val = "\n".join([f"{i} - {a}" for i, a in enumerate(symp_pred.cpu().numpy().T.tolist())])
                # print(f"Turn {curr_turn} - symp_pred: \n {tmp_val}")
                if masked_inquired_actions:
                    symp_pred = symp_pred.masked_fill(mask.bool(), float("-inf"))

                sym_action = torch.argmax(symp_pred, dim=-1).cpu().numpy()
                # print(f"Turn {curr_turn} - sym_action: {sym_action.tolist()}")
                sym_index = sym_action

                if masked_inquired_actions:
                    mask[range(obs.shape[0]), sym_action.tolist()] = 1

                prev_agent_act[~done] = 1
                prev_user_act[~done] = 2 - self.env.all_state[~done, sym_index[~done]]

                obs, _, _ = self.env.step(obs, sym_action, done)
                valid_timesteps[-1][~done] = True 
                curr_turn += 1

            if any(prev_agent_act == 1):
                inputs = np.zeros(
                    (obs.shape[0], obs.shape[1] + num_agent_actions + num_user_actions)
                )
                inputs[:, :user_offset] = obs
                inputs[range(obs.shape[0]), user_offset + prev_user_act] = 1
                inputs[range(obs.shape[0]), agent_offset + prev_agent_act] = 1
                inputs = torch.from_numpy(inputs).float().to(device)

                # import pdb;pdb.set_trace()
                _, _, patho_pred = agent(inputs, inputs)
                patho_pred = (
                    None
                    if patho_pred is None
                    else F.softmax(patho_pred, dim=-1).detach().view(obs.shape[0], -1).cpu().numpy()
                )
                all_diags.append(patho_pred if patho_pred is not None else (np.ones((obs.shape[0], self.env.diag_size))/self.env.diag_size))
                final_diagnosis[~done] = all_diags[-1][~done]
                
            valid_timesteps = np.array(valid_timesteps).swapaxes(0, 1)
            all_diags = np.array(all_diags).swapaxes(0, 1)
            
            ave_step = np.sum(valid_timesteps, axis=-1) + 1
            ave_pos = np.sum(self.env.inquired_symptoms * self.env.all_state, axis=-1) / np.maximum(1, ave_step)
            
            overall_il += np.sum(ave_step)
            overall_recall += np.sum(ave_pos)
                
            if compute_metrics_flag:
                batch_metrics = compute_metrics(
                    self.env.target_differential, true_disease,
                    final_diagnosis, all_diags, valid_timesteps,
                    self.env.all_state, self.env.inquired_symptoms,
                    self.env.symptom_mask, self.env.atcd_mask, self.env.severity_mask, tres=0.01
                )
                tmp_n = batch_idx * batch_size
                tmp_m = obs.shape[0]
                tmp_t = tmp_n + obs.shape[0]
                all_metrics = {a: (tmp_n / (tmp_t)) * all_metrics.get(a, 0) + (tmp_m / (tmp_t)) * batch_metrics[a] for a in batch_metrics.keys()}

        results_dict = {
            "average_evidence_recall": overall_recall / max(1, len(eval_patient_ids)),
            "average_step": overall_il / max(1, len(eval_patient_ids)),
        }
        if compute_metrics_flag:
            all_metrics = {a: all_metrics[a].tolist() if hasattr(all_metrics[a], "tolist") else all_metrics[a] for a in all_metrics.keys()}
            results_dict.update(all_metrics)        

        return results_dict
