class TrainStages(list):
    def __init__(self, is_pretrain: bool, is_train_csqvae: bool, is_finetuning: bool):
        self.stages = ["sqvae", "diffusion", "classification", "csqvae", "finetuning"]
        self.is_pretrain = is_pretrain
        self.is_train_csqvae = is_train_csqvae
        self.is_finetuning = is_finetuning

    def get_train_flag(self, train_stage: str):
        if train_stage in self.stages[:3]:
            return self.is_pretrain
        elif train_stage == self.stages[3]:
            return self.is_train_csqvae
        elif train_stage == self.stages[4]:
            return self.is_finetuning
        else:
            raise AttributeError

    def __len__(self):
        return len(self.stages)

    def __getitem__(self, index: int):
        train_stage = self.stages[index]
        train_flag = self.get_train_flag(train_stage)
        return train_stage, train_flag

    def __iter__(self):
        for train_stage in self.stages:
            train_flag = self.get_train_flag(train_stage)
            yield train_stage, train_flag

    def __contains__(self, key):
        return key in self.stages
