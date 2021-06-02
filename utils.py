from datetime import datetime


def gettime():
    return datetime.now().strftime("%m-%d-%Y %H:%M")