# Requisitos para correr un modelo. El mensaje es para el usuario: se muestra sin traceback.


class ModelError(Exception):
    pass


class CheckpointError(ModelError):  # no está, está dañado o no es de este programa
    pass


class DependencyError(ModelError):  # falta un paquete o un archivo de la arquitectura
    pass


class DeviceError(ModelError):  # el dispositivo no existe o no corre el modelo
    pass
