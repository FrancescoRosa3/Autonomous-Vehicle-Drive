class Obstacle:

    def __init__(self, x, y, z, yaw, xrad, yrad, zrad, id_class):
        self.x = x
        self.y = y
        self.z = z
        self.yaw = yaw * pi / 180   #from deg to rad
        self.xrad = xrad
        self.yrad = yrad
        self.zrad = zrad
        self.id_class = id_class

    def set_position(self, x, y, z, yaw, xrad, yrad, zrad):
        self.x = x
        self.y = y
        self.z = z
        self.yaw = yaw * pi / 180   #from deg to rad
        self.xrad = xrad
        self.yrad = yrad
        self.zrad = zrad

    def get_position(self):
        return self.x, self.y, self.z, self.yaw, self.xrad, self.yrad, self.zrad
    
    def set_id_class(self, id_class):
        self.id_class = id_class

    def set_id_class(self):
        return self.id_class