#@dataclass
class GymPacket:
    def __init__(self, action, data):
        self.data = data
        self.action = action
