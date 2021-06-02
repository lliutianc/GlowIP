from datetime import datetime


def gettime():
    return datetime.now().strftime("%d-%m-%Y %H:%M")