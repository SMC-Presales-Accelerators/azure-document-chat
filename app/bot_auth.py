import requests
import os

def bot_auth():
    BOT_DIRECTLINE_SECRET_KEY = os.environ['BOT_DIRECTLINE_SECRET_KEY']
    headers = {'Authorization': "Bearer %s" % BOT_DIRECTLINE_SECRET_KEY}
    r = requests.post("https://directline.botframework.com/v3/directline/tokens/generate", headers=headers)
    return r.json()
    