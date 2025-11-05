import os
import pickle

class FileUtil:
    @staticmethod
    def loadmodel(filename):
        try:
            path = os.path.abspath(filename)
            print(">>> Loading model from:", path)
            with open(path, 'rb') as f:
                model = pickle.load(f)
            print(">>> Model loaded successfully!")
            return model
        except Exception as e:
            print(">>> Error loading model:", e)
            return None
