import json
import random

import pandas as pd


SEG1 = "_@_"
SEG2 = "|||"

class Patient:
    def __init__(self, record):
        assert len(record) == 6
        self.age = record[0]
        self.sex = record[2]
        self.pathology = record[3]
        self.init_evidence = record[5]
        self.follow_evidences = eval(record[4])


CHOICES = {"0": "No", "1": "Yes", "N": "No", "Y": "Yes", "F": "Female", "M": "Male"}
class Evidences:
    def __init__(self, fpath):
        self._evids = json.load(open(fpath, encoding="utf8"))
        for i, name in enumerate(sorted(self._evids.keys())):
            evid = self._evids[name]
            self._evids[i] = name
            evid["Index"] = i
            if evid["data_type"] == "B":
                evid["value_meaning"] = {"0": {"en": "No"}, "1": {"en": "Yes"}}

            if evid["data_type"] == "C" and isinstance(evid["default_value"], int):
                evid["value_meaning"] = {}
                minval = min(evid["possible-values"])
                maxval = max(evid["possible-values"])
                prefix = "In a scale between %s to %s, it is level " % (minval, maxval)
                for val in evid["possible-values"]:
                    evid["value_meaning"][str(val)] = {"en": prefix + str(val)}

            evid["choices"] = {}
            for x, y in evid.pop("value_meaning").items():
                evid["choices"][x] = CHOICES.get(y["en"], y["en"])

            for key in ["code_question", "question_fr", "data_type",
                        "is_antecedent", "possible-values"]:
                del evid[key]

    def __len__(self):
        return len(self._evids) // 2

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self._conds[self._conds[idx]]
        return self._conds[idx]

    def groupby_evidence(self, evids):
        new_evids = []
        last_name, cached_val = None, []
        for name in evids:
            val = "1"
            if SEG1 in name:
                name, val = name.split(SEG1)
            if name != last_name:
                if last_name != None:
                    new_evids.append(last_name + SEG1 + SEG2.join(cached_val))
                last_name, cached_val = name, [val]
        new_evids.append(last_name + SEG1 + SEG2.join(cached_val))
        return new_evids

    def get_symptom(self, symp):
        symp, val = symp.split(SEG1)
        symp = self._evids[symp]
        if val == "":
            val = str(evid["default_value"])
        answers = [symp["choices"][_] for _ in val.split(SEG2)]
        if len(answers) == 1:
            answers = answers[0]
        elif len(answers) == 2:
            answers = " and ".join(answers)
        else:
            answers = ", ".join(answers[:-1]) + ", and " + answers[-1]
        if not answers.endswith("."):
            answers += "."
        return {"name": symp["name"],
                "question": symp["question_en"],
                "answer": answers}

    def get_basic_info(self, patient):
        initial = self.groupby_evidence([patient.init_evidence])
        initial = self.get_symptom(initial[0])["question"]
        report = initial.split(" ", 1)[1]
        for l, r in [("the person", ""), ("your", "my"), ("?", "."),
                     ("you", "I"), ("any", "some"), ("or", "and"),
                     ("discomfandt", "discomfort")]:
            report = report.replace(l, r)
        prompt = "Patient Information\n" +\
                 "-------------------\n" +\
                 "Sex: %s\nAge: %s\n" % (CHOICES[patient.sex], patient.age) +\
                 "Self-Report: %s\n" % report.strip().capitalize()
        return prompt

    def get_conversation(self, patient):
        sympotems = self.groupby_evidence(patient.follow_evidences)
        prompt = "Inquiries Dialogue\n" +\
                 "---------------------\n"
        for evid in sympotems:
            if evid.split(SEG1)[0] == patient.init_evidence:
                continue
            evid = self.get_symptom(evid)
            prompt += "Doctor: %s\n" % evid["question"].capitalize()
            prompt += "Patient: %s\n" % evid["answer"].capitalize()
        return prompt

    def simulate_conversation(self, patient):
        basic = self.get_basic_info(patient)
        conv = self.get_conversation(patient)
        return basic + '\n' + conv


class Conditions:
    def __init__(self, fpath):
        self._conds = json.load(open(fpath, encoding="utf8"))
        for i, name in enumerate(sorted(self._conds.keys())):
            self._conds[name]["Index"] = i
            self._conds[name]["symptoms"] = sorted(self._conds[name]["symptoms"])
            del self._conds[name]["cond-name-fr"]
            self._conds[i] = name

    def __len__(self):
        return len(self._conds) // 2

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self._conds[self._conds[idx]]
        return self._conds[idx]

    def get_diagnosis(self, patient):
        return self[patient.pathology]["Index"]


class PatientsDataset:
    def __init__(self, data_path, evid_path, cond_path):
        self.conditions = Conditions(cond_path)
        self.evidences = Evidences(evid_path)
        self.patients = []
        for row in pd.read_csv(data_path).to_numpy().tolist():
            self.patients.append(Patient(row))

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        patient = self.patients[idx]
        conversation = self.evidences.simulate_conversation(patient)
        return conversation, patient.pathology


def encode(string):
    string = string.strip().replace("\r", "")
    string = string.replace("\n", "\\n")
    string = string.replace("\t", "\\t")
    return string


if __name__ == "__main__":
    for subset in ["train", "validate", "test"]:
        dataset = PatientsDataset("release_%s_patients.csv" % subset,
                                  "release_evidences.json",
                                  "release_conditions.json")
        with open("./ddxplus_%s.tsv" % subset, 'w', encoding="utf8") as f:
            for i in range(len(dataset)):
                text, diag = dataset[i]
                f.write(encode(text) + '\t' + str(diag) + '\n')
