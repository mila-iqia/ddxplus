import ast
import json
import warnings

import numpy as np
import pandas as pd


def clean(data):
    """Utility function for cleaning pathology name for csv (column) purposes.

    Replaces commas and line breaks in the source string with a single space.

    Parameters
    ----------
    data: str
        data string to be cleaned.

    Returns
    -------
    result: str
        the resulting string.

    """
    result = data.replace("\r\n", " ")
    result = result.replace("\r", " ")
    result = result.replace("\n", " ")
    result = result.replace(",", " ")
    return result


def preprocess_symptoms(symptoms):
    """Utility function for pre-processing symptom data.

    Preprocess symptoms data as provided by the csv file from the Synthea generation.

    Parameters
    ----------
    symptoms: str
        String representing the symptoms as exported by Synthea.
        Ex. : Nasal congestion:30;Sore throat:42;Vomiting:35

    Returns
    -------
    result: list
        a list of all the symptoms.

    """
    if pd.isnull(symptoms):
        return []

    # symptoms is a string of list
    if symptoms.startswith("[") and symptoms.endswith("]"):
        return ast.literal_eval(symptoms)

    data = symptoms.split(";")
    result = []
    for x in data:
        # here we remove the severity of the symptoms as it is not
        # needed for this project
        name = x.split(":")[0]
        result.append(name)

    # sort result
    result = sorted(result)
    return result


def preprocess_differential(differential):
    """Utility function for pre-processing differential data.

    Preprocess differential data as provided by the csv file
    obtained when evaluating the synthesised patients through DXA.

    Parameters
    ----------
    differential : str
        String representing the differential as exported from DXA.
        The format is patho1:sommeOR1:score1;patho2:sommeOR2:score2
        Eg.: Laryngite aigue:0.220:1.096;Angine instable:0.205:1.189

    Returns
    -------
    result: list
        a list of all the pathologies involved in the differential
        together with their respective scores.

    """
    if pd.isnull(differential):
        return []

    # differential is a string of list
    if differential.startswith("[") and differential.endswith("]"):
        return ast.literal_eval(differential)

    data = differential.split(";")
    result = []
    for x in data:
        # here we retrieve the scores associated to each pathology.
        split = x.split(":")
        # some pathology may have the : sign in it
        # e.g: 'ddx : vaginite atrophique  hyperplasie/néo endomètre  fibrome ..'
        split[0] = ":".join(split[:-2])
        split[1] = float(split[-2])
        split[2] = float(split[-1])
        split = split[0:3]
        result.append(split)

    # sort result
    result = sorted(result, key=lambda x: x[0])
    return result


def only_contain_derivated_symptoms(symptoms):
    """Utility function to check the validity of symptom data.

    Determined if symptoms data as provided only contain
    derivated symptoms i.e. symptoms with '_@_' string.

    A derivated symptom is a question/symptom associated with
    properties from the DXA knowledge database derived from
    `code_question` having non_empty `body parts` (endroits)
    characteristics. Eg. douleurxx_intens_@_1.

    Parameters
    ----------
    symptoms: list
        List of all symptoms.

    Returns
    -------
    result: bool
        True if the provided symptoms only contain derivated symptoms,
        False otherwise.

    """
    for a in symptoms:
        if not ("_@_" in a):
            return False
    return True


def get_symptoms_with_multiple_answers(symptoms):
    """Utility function to retrieve symptoms with multiple entries.

    Parameters
    ----------
    symptoms : list
        List of all symptoms.

    Returns
    -------
    result: list
        a list of symptoms with multiple answers/entries.

    """
    tmp = []
    for a in symptoms:
        idx = a.find("_@_")
        if idx == -1:
            tmp.append(a)
        else:
            tmp.append(a[:idx])
    count = {}
    result = []
    for a in tmp:
        if a not in count:
            count[a] = 1
        else:
            if count[a] == 1:
                result.append(a)
            count[a] += 1
    return result


def convert_to_compatible_format(df):
    """Utility function for converting relased csv format into legacy format.
    Parameters
    ----------
    df : DataFrame
        The dataframe in the released format.
    Returns
    -------
    result: DataFrame
        The dataframe in the legacy format.
    """
    # get original names
    df = df.rename(
        columns={
            "AGE": "AGE_BEGIN",
            "DIFFERENTIAL_DIAGNOSIS": "DIFFERNTIAL_DIAGNOSIS",
            "SEX": "GENDER",
            "EVIDENCES": "SYMPTOMS",
        }
    )
    # add NUM_SYMPTOMS (fake data)
    num_symptoms = [5] * len(df)
    df["NUM_SYMPTOMS"] = num_symptoms
    # add Ethniticy (fake data) - not used in practice when training agent
    n = len(df) // 2
    ethnicity = ["nonhispanic"] * n + ["hispanic"] * (len(df) - n)
    df["ETHNICITY"] = ethnicity
    # add FOUND_GT_PATHOLOGY (fake data)
    found_gt = [True] * len(df)
    df["FOUND_GT_PATHOLOGY"] = found_gt
    if "RACE" not in df.columns:
        # add fake race data - not used in practice
        m = len(df) // 5
        race = ["white", "black", "asian", "native", "other"] * m + ["black"] * max(0, (len(df) - 5 * m))
        race = race[0:len(df)]
        df["RACE"] = race
    # process differential diagnosis
    df["DIFFERNTIAL_DIAGNOSIS"] = df["DIFFERNTIAL_DIAGNOSIS"].apply(
        lambda x: stringify_differential(x)
    )
    if "INITIAL_EVIDENCE" in df.columns:
        df = df.rename(columns={"INITIAL_EVIDENCE": "INITIAL_SYMPTOM"})
    return df


def convert_to_compatible_json_format(data):
    """Utility function for converting relased json data into legacy format.
    Parameters
    ----------
    data : dict
        The json data in the released format.
    Returns
    -------
    result: dict
        The json data in the legacy format.
    """
    # replace "data_type" by "type-donnes" when possible
    # replace "severity" by "urgence" when possible
    for x in data.keys():
        if "data_type" in data[x]:
            data[x]["type-donnes"] = data[x]["data_type"]
            data[x].pop("data_type")
        if "severity" in data[x]:
            data[x]["urgence"] = data[x]["severity"]
            data[x].pop("severity")
    return data


def stringify_differential(differential):
    """Utility function for stringifying differential data.
    Stringify differential data as provided by the released csv file.
    Parameters
    ----------
    differential : list
        List representing the differential data.
        The format is [[patho1, prob1], [patho2, prob2], ...].
    Returns
    -------
    result: str
        the string version of the differential data
        together with their respective scores.
        The format is patho1:sommeOR1:score1;patho2:sommeOR2:score2
        Eg.: Laryngite aigue:0.220:1.096;Angine instable:0.205:1.189
    """
    if (differential is None) or len(differential) == 0:
        return None
    # differential is a string of list
    is_str = isinstance(differential, str)
    if is_str and differential.startswith("[") and differential.endswith("]"):
        differential = ast.literal_eval(differential)
    result = []
    for x in differential:
        # here we retrieve the patho and proba.
        assert len(x) == 2, f"{x} => {differential}"
        patho, proba = x
        sommeOR = proba / (1 - proba) if proba != 1 else 100
        score = "1"  # fake data
        data = ":".join([patho, str(sommeOR), score])
        # some pathology may have the : sign in it
        result.append(data)
    # sort result
    final_str = ";".join(result)
    return final_str


def load_csv(filepath, map_to_ids=True):
    """Utility function to load and pre-process a Synthea generated patient csv file.

    As for pre-processing, it consists of:
        - removal of rows with zero symptoms
        - removal of rows containing only derivated symptoms
        - pre-processing of symptom data to transform them from string to list
        - retrieval of unique pathologies present in the file
        - retrieval of unique symptoms present in the file
        - retrieval of unique symptoms per pathology present in the file
        - retrieval of unique races present in the file
        - retrieval of unique ethnics present in the file
        - retrieval of unique genders present in the file

    Parameters
    ----------
    filepath: str
        path to the csv file containing generated patients
        from Synthea simulator.
    map_to_ids: bool
        Bool indicating if the columns PATHOLOGY, SYMPTOMS, ETHNICITY,
        RACE and GENDER in the dataframe need to be converted to ids.
        Default: True

    Returns
    -------
    df: DataFrame
        a pandas data frame containing the simuated patients.
    unique_symptoms: list
        a list of unique symptoms within the provided data.
    unique_pathologies: list
        a list of unique pathologies within the provided data.
    pathology_symptoms: dict
        a mapping from a pathology to a set of symptoms describing that
        pathology as derived from the dataframe.
    unique_race: list
        a list of unique races within the provided data.
    unique_ethnics: list
        a list of unique ethnics within the provided data.
    unique_genders: list
        a list of unique genders within the provided data.
    symptoms_with_multiple_answers: list
        a list of symptoms accepting multiple answers.
    max_differential_len: int
        the maximum number of pathologies present in the differential
        diagnosis. If the differntial are not provided within the data,
        then the returned value is -1.
    unique_differential_pathos: list
        a list of pathologies that are part of differential diagnosis if any.
    unique_init_symptom: list
        a list of unique initial symptoms within the provided data.

    """

    def merge_data(x):
        return set([a for b in x.tolist() for a in b])

    df1 = pd.read_csv(filepath, sep=",")

    # convert to legacy format if needed
    df = convert_to_compatible_format(df1) if ("EVIDENCES" in df1.columns) else df1

    # we remove patient with zero symptoms
    df = df[df["NUM_SYMPTOMS"] != 0].reset_index(drop=True)

    # we parse the symptoms to remove the severity level as we do
    # not need it for this project
    df["SYMPTOMS"] = df["SYMPTOMS"].apply(lambda x: preprocess_symptoms(x))

    # remove rows that only contain derivated symptoms
    df = df[~df["SYMPTOMS"].map(only_contain_derivated_symptoms)].reset_index(drop=True)

    unique_pathologies = sorted(df["PATHOLOGY"].unique().tolist())

    df2 = df.groupby(["PATHOLOGY"]).agg({"SYMPTOMS": merge_data})

    unique_symptoms = df2.SYMPTOMS.agg(merge_data)
    unique_symptoms = sorted(list(unique_symptoms))

    pathology_symptoms = df2.to_dict()["SYMPTOMS"]

    is_initial_symptom_present = "INITIAL_SYMPTOM" in df.columns

    # `FIRST_SYMPTOM` column is only present in the data meant for
    # hand annotation.
    all_columns = [
        "PATIENT",
        "GENDER",
        "RACE",
        "ETHNICITY",
        "AGE_BEGIN",
        "AGE_END",
        "PATHOLOGY",
        "NUM_SYMPTOMS",
        "SYMPTOMS",
        "DIFFERNTIAL_DIAGNOSIS",
        "FOUND_GT_PATHOLOGY",
        "INITIAL_SYMPTOM",
        "FIRST_SYMPTOM",
    ]
    to_drop = set(df.columns) - set(all_columns)
    to_drop = list(to_drop) + ["PATIENT", "AGE_END"]
    if not map_to_ids:
        to_drop.remove("PATIENT")
    df = df.drop(columns=to_drop, errors="ignore")

    symptoms_with_multiple_answers = (
        df["SYMPTOMS"]
        .apply(lambda x: get_symptoms_with_multiple_answers(x))
        .agg(merge_data)
    )
    symptoms_with_multiple_answers = sorted(list(symptoms_with_multiple_answers))

    unique_races = sorted(df["RACE"].unique().tolist())
    unique_ethnics = sorted(df["ETHNICITY"].unique().tolist())
    unique_gender = sorted(df["GENDER"].unique().tolist())
    unique_init_sympt = (
        []
        if not is_initial_symptom_present
        else sorted(df["INITIAL_SYMPTOM"].unique().tolist())
    )

    dict_patho = {a: i for i, a in enumerate(unique_pathologies)}
    dict_symptom = {a: i for i, a in enumerate(unique_symptoms)}
    dict_races = {a: i for i, a in enumerate(unique_races)}
    dict_ethnics = {a: i for i, a in enumerate(unique_ethnics)}
    dict_gender = {a: i for i, a in enumerate(unique_gender)}
    dict_init_sympt = {a: i for i, a in enumerate(unique_init_sympt)}

    if map_to_ids:
        df["PATHOLOGY"] = df["PATHOLOGY"].apply(lambda x: dict_patho[x])
        df["SYMPTOMS"] = df["SYMPTOMS"].apply(lambda x: [dict_symptom[a] for a in x])
        df["ETHNICITY"] = df["ETHNICITY"].apply(lambda x: dict_ethnics[x])
        df["RACE"] = df["RACE"].apply(lambda x: dict_races[x])
        df["GENDER"] = df["GENDER"].apply(lambda x: dict_gender[x])
        df["INITIAL_SYMPTOM"] = (
            [-1] * len(df)
            if not is_initial_symptom_present
            else df["INITIAL_SYMPTOM"].apply(lambda x: dict_init_sympt[x])
        )

    pathology_symptoms = {
        dict_patho[a]: set([dict_symptom[b] for b in pathology_symptoms[a]])
        for a in pathology_symptoms.keys()
    }

    # define max differential length
    max_differential_len = -1

    # define unique patho involved in differential diagnosis
    unique_differential_pathos = []

    # in case the differential diagnosis is present
    if "DIFFERNTIAL_DIAGNOSIS" in df.columns:
        assert "FOUND_GT_PATHOLOGY" in df.columns

        # filter to keep only entries whose differential is not empty
        df = df[df.DIFFERNTIAL_DIAGNOSIS.notnull()].reset_index(drop=True)

        # filter to keep only entries whose patho is part of the differential
        df = df[df.FOUND_GT_PATHOLOGY].reset_index(drop=True)

        # preprocessing differential
        df["DIFFERNTIAL_DIAGNOSIS"] = df["DIFFERNTIAL_DIAGNOSIS"].apply(
            lambda x: preprocess_differential(x)
        )

        # encode the differential
        # here we exclude pathos that are not part of the simulated ones
        differential_pathos = (
            df["DIFFERNTIAL_DIAGNOSIS"]
            .apply(lambda x: [a[0] for a in x])
            .agg(merge_data)
        )
        unique_differential_pathos = sorted(list(differential_pathos))
        dict_differential = {a: i for i, a in enumerate(unique_differential_pathos)}
        if map_to_ids:
            df["DIFFERNTIAL_DIAGNOSIS"] = df["DIFFERNTIAL_DIAGNOSIS"].apply(
                lambda x: [[dict_differential[a[0]], a[1], a[2]] for a in x]
            )

        # get the max diffential length
        dif_len = df.DIFFERNTIAL_DIAGNOSIS.map(len)
        max_differential_len = dif_len.max()
        assert (dif_len > 0).all()

    return_values = [
        df,
        unique_symptoms,
        unique_pathologies,
        pathology_symptoms,
        unique_races,
        unique_ethnics,
        unique_gender,
        symptoms_with_multiple_answers,
        max_differential_len,
        unique_differential_pathos,
        unique_init_sympt,
    ]

    return return_values


def load_and_check_data(data_filepath, provided_data, key_name):
    """Utility function to load and check the validity of condition/symptom JSON files.

    It loads the data from the JSON file and check if the provided data are
    compliant with those.

    Parameters
    ----------
    symptom_filepath :  str
        path to a json file containing the authorized symptom data.
        the minimum structure of the data should be:

        .. code-block:: text

            {
                key_data1: {
                    key_name: data-name1,
                    ...
                },
                key_data2: {
                    key_name: data-name2,
                    ...
                },
                ...
            }

    provided_data : list
        list of syptoms as provided by the data from Synthea patient
        generation.
    key_name : str
        the key used to access the information of the same meaning
        as the one in `provided_data`.

    Returns
    -------
    index_2_key: list
        a list containing all the keys of the authorized data.
    name_2_index: dict
        a dict mapping the name associated to the authorized data to an index.
    data: dict
        the authorized data.

    """

    with open(data_filepath) as fp:
        data = json.load(fp)

    data = convert_to_compatible_json_format(data)
    index_2_key = sorted(list(data.keys()))
    for k in index_2_key:
        data[k][key_name] = clean(data[k][key_name])
    name_2_index = {data[index_2_key[i]][key_name]: i for i in range(len(index_2_key))}

    data_names = [data[k][key_name] for k in index_2_key]
    is_present = []
    for elem in provided_data:
        if elem in data_names:
            is_present.append(True)
        else:
            # it is a derivated symptom. therefore it must contains _@_
            # substring.
            idx = elem.find("_@_")
            if idx == -1:
                is_present.append(False)
            else:
                # a deriveted symptom is formed by `base + _@_ + value` where
                # `value` is the value associated to the symptom and the `base` is
                # the symptom itself as supposely provided in the JSON file.
                elem_base = elem[:idx]
                elem_val = elem[idx + 3 :]
                base_idx = name_2_index.get(elem_base, -1)
                if base_idx == -1:
                    is_present.append(False)
                possible_values = data.get(index_2_key[base_idx], {}).get(
                    "possible-values", []
                )
                possible_values = [str(a) for a in possible_values]
                if elem_base in data_names and str(elem_val) in possible_values:
                    is_present.append(True)
                else:
                    is_present.append(False)

    has_all_data = all(is_present)

    if not has_all_data:
        indices = [i for i, v in enumerate(is_present) if not v]
        raise ValueError(
            "The provided symptom samples are not compliant with "
            + "authorized symptoms in the json file: {} : {}".format(
                data_filepath, [provided_data[i] for i in indices]
            )
        )

    return index_2_key, name_2_index, data


def encode_age(age):
    """Utility function for encoding the `age`.

    It encodes the age according to the Symcat age bins:
        - [0, 1[
        - [1, 4]
        - [5, 14]
        - [15, 29]
        - [30, 44]
        - [45, 59]
        - [60, 74]
        - 75+

    Parameters
    ----------
    age :  int
        age of the patient

    Returns
    -------
    category: int
        the category (bin) in which the age belongs to
    """
    if age < 0:
        raise ValueError("Age could not less than 0!")
    if age < 1:
        return 0
    elif age >= 1 and age <= 4:
        return 1
    elif age >= 5 and age <= 14:
        return 2
    elif age >= 15 and age <= 29:
        return 3
    elif age >= 30 and age <= 44:
        return 4
    elif age >= 45 and age <= 59:
        return 5
    elif age >= 60 and age <= 74:
        return 6
    else:
        return 7


def encode_sex(sex):
    """Utility function for encoding the `sex`.

    It encodes the sex of the patient:
        - 0 for "M" and 1 for "F"

    Parameters
    ----------
    sex :  str
        sex of the patient

    Returns
    -------
    category: int
        the corresponding category of the provided data
    """
    if sex == "M":
        return 0
    elif sex == "F":
        return 1
    else:
        raise ValueError("Unexpected sex value: " + sex)


def decode_sex(code):
    """Utility function for decoding the sex `code`.

    It decodes the sex of the patient:
        - 0 for "M" and 1 for "F"

    Parameters
    ----------
    code :  int
        sex code of the patient

    Returns
    -------
    sex: str
        the corresponding sex of the provided data
    """
    if code == 0:
        return "M"
    elif code == 1:
        return "F"
    else:
        raise ValueError("Unexpected sex code value: " + code)


def encode_race(race):
    """Utility function for encoding the `race`.

    It encodes the race of the patient.
        - 0 for "white"
        - 1 for "black"
        - 2 for "asian"
        - 3 for "native"
        - 4 otherwise.

    Parameters
    ----------
    race :  str
        race of the patient

    Returns
    -------
    category: int
        the corresponding category of the provided data
    """
    if race == "white":
        return 0
    elif race == "black":
        return 1
    elif race == "asian":
        return 2
    elif race == "native":
        return 3
    else:
        return 4


def encode_ethnicity(ethnic):
    """Utility function for encoding the `ethnic`.

    It encodes the ethnic of the patient.
        - 0 for "hispanic" and 1 for "nonhispanic"

    Parameters
    ----------
    ethnic :  str
        ethnic of the patient

    Returns
    -------
    category: int
        the corresponding category of the provided data
    """
    if ethnic == "hispanic":
        return 0
    elif ethnic == "nonhispanic":
        return 1
    else:
        raise ValueError("Unexpected ethnic value: " + ethnic)


def encode_geo(geo):
    """Utility function for encoding the geographical region `geo`.

    It encodes the geographical region of the patient.

    Parameters
    ----------
    geo :  str
        geographical region of the patient

    Returns
    -------
    category: int
        the corresponding category of the provided data
    """
    regio_dict = {
        "N": 0,
        "AfriqN": 1,
        "AfriqO": 2,
        "AfriqSS": 3,
        "AmerC": 4,
        "AmerN": 5,
        "AmerS": 6,
        "Asie": 7,
        "AsieSSE": 8,
        "Cara": 9,
        "Euro": 10,
        "Ocea": 11,
    }
    if geo in regio_dict:
        return regio_dict[geo]
    else:
        raise ValueError("Unexpected Geo value: " + geo)


def decode_geo(code):
    """Utility function for decoding the geographical region given the `code`.

    Parameters
    ----------
    code : int
        the code associated with geographical region

    Returns
    -------
    geo: str
        the corresponding geographical region
    """
    regio_dict = {
        0: "N",
        1: "AfriqN",
        2: "AfriqO",
        3: "AfriqSS",
        4: "AmerC",
        5: "AmerN",
        6: "AmerS",
        7: "Asie",
        8: "AsieSSE",
        9: "Cara",
        10: "Euro",
        11: "Ocea",
    }
    if code in regio_dict:
        return regio_dict[code]
    else:
        raise ValueError("Unexpected Geo code value: " + str(code))


def convert_lst_to_str(lst, prefix, suffix):
    """Converts input list to string.

    This function appends the specified prefix and suffix to
    each element of the input list and converts the list into
    a string.

    Parameters
    ----------
    lst:  list
        A list of elements that will be converted to a string.
    prefix: str
        A prefix that needs to be added to each element of the
        the input list.
    suffix: str
        A suffix that needs to be added to each element of the
        the input list.

    Returns
    -------
    result: str
        A string containing elements of the input list with the
        prefix and suffix added to the elements.

    """
    return "".join([f"{prefix}{elem}{suffix}" for elem in lst])


def convert_patho_idx_to_patho_str(
    indices_and_probas, idx_to_patho_map, prefix, suffix
):
    """Converts the input list of pairs to a string.

    This function converts the list of input pathology index and probability
    pairs to a multi-line string.

    Parameters
    ----------
    indices_and_probas:  list
        A list containing pair of differential diagnosis pathology
        indices and their probabilities.
    idx_to_patho_map: list
        An array containing the differential diagnosis
        pathologies.
    prefix: str
        A prefix that needs to be added to each element of the
        the input list.
    suffix: str
        A suffix that needs to be added to each element of the
        the input list.

    Returns
    -------
    patho_names_with_proba_str: str
        A string containing elements of the input list with the
        prefix and suffix added to the elements.

    """
    if not isinstance(indices_and_probas, list):
        indices_and_probas = list(indices_and_probas)
    indices_and_probas.sort(key=lambda x: x[1], reverse=True)
    patho_names_with_proba = [
        f"{idx_to_patho_map[idx]}: {proba}"
        for idx, proba in indices_and_probas
        if idx != -1
    ]  # -1 index is null index
    patho_names_with_proba_str = convert_lst_to_str(
        patho_names_with_proba, prefix, suffix
    )
    return patho_names_with_proba_str
