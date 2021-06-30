class LinearSchedule:
    def __init__(self, base, limit, n_iter):
        self.c = 0
        self.b = base
        self.l = limit
        self.t = n_iter
    def __call__(self):
        self.c += 1
        return min(self.l, self.b + self.c * (self.l - self.b) / self.t)
    def get_ls(self):
        return min(self.l, self.b + self.c * (self.l - self.b) / self.t)
