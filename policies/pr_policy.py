from numpy.random import rand

class PRPolicy():
    def __init__(self, protagonist, adversary, epsilon=0.1):
        self.protagonist = protagonist
        self.adversary = adversary
        self.epsilon = epsilon

    def action_eval_np(self, obs, **kwargs):
        return self.protagonist.action_eval_np(obs, **kwargs)

    def get_policy(self):
        return self.protagonist if rand() > self.epsilon else self.adversary

    def action_np(self, obs, **kwargs):
        return self.get_policy().action_np(obs, **kwargs)
    
    def action(self, obs, **kwargs):
        raise NotImplementedError
        # return self.get_policy().action(obs, **kwargs)



    def reset(self):
        self.protagonist.reset()
        self.adversary.reset()

    def get_snapshot(self):
        # return self.state_dict()
        protagonist_snapshot = self.protagonist.get_snapshot()
        adversary_snapshot = self.adversary.get_snapshot()
        adversary_snapshot = {f"adversary/{key}": value for key, value in adversary_snapshot.items()}
        protagonist_snapshot.update(adversary_snapshot)
        return protagonist_snapshot

    def load_snapshot(self, state_dict):
        # self.load_state_dict(state_dict)
        raise NotImplementedError

    def get_diagnostics(self):
        return {}

    def has_nan(self):
        return self.protagonist.has_nan() or self.adversary.has_nan()

    def train(self, mode):
        self.protagonist.train(mode)
        self.adversary.train(mode)