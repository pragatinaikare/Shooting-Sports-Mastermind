import requests
import threading
import sys
import json
import os
import time


def load_keys():
    f = open("Docs/keys.txt")
    k = "".join(f.readlines())
    keys = json.loads(k)
    f.close()
    return keys
