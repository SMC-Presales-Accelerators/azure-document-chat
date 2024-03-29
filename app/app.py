# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import sys
import traceback
from datetime import datetime

from aiohttp import web
from aiohttp.web import Request, Response, json_response
from botbuilder.core import (
    BotFrameworkAdapterSettings,
    TurnContext,
    BotFrameworkAdapter,
    ShowTypingMiddleware,
)
from botbuilder.core.integration import aiohttp_error_middleware
from botbuilder.schema import Activity, ActivityTypes

from bot import MyBot

from bot_auth import bot_auth

from search_index import AzureSearchApi

from dotenv import load_dotenv
load_dotenv()

PORT = 3978
APP_ID = os.environ.get("MICROSOFT_APP_ID", "")
APP_PASSWORD = os.environ.get("MICROSOFT_APP_PASSWORD", "")

BASE_PATH = os.environ.get("DOC_CHAT_BASE_PATH", "")
if BASE_PATH != "" and BASE_PATH[-1] == "/":
    BASE_PATH = BASE_PATH[:-1]
if BASE_PATH != "" and BASE_PATH[0] != "/":
    BASE_PATH = "/" + BASE_PATH

# Create adapter.
# See https://aka.ms/about-bot-adapter to learn more about how bots work.
SETTINGS = BotFrameworkAdapterSettings(APP_ID, APP_PASSWORD)
ADAPTER = BotFrameworkAdapter(SETTINGS)

# Use Azure AI Search to get titles for sidebar
search = AzureSearchApi()


# Catch-all for errors.
async def on_error(context: TurnContext, error: Exception):
    # This check writes out errors to console log .vs. app insights.
    # NOTE: In production environment, you should consider logging this to Azure
    #       application insights.
    print(f"\n [on_turn_error] unhandled error: {error}", file=sys.stderr)
    traceback.print_exc()

    # Send a message to the user
    await context.send_activity("The bot encountered an error or bug.")
    await context.send_activity(
        "To continue to run this bot, please fix the bot source code."
    )
    # Send a trace activity if we're talking to the Bot Framework Emulator
    if context.activity.channel_id == "emulator":
        # Create a trace activity that contains the error object
        trace_activity = Activity(
            label="TurnError",
            name="on_turn_error Trace",
            timestamp=datetime.utcnow(),
            type=ActivityTypes.trace,
            value=f"{error}",
            value_type="https://www.botframework.com/schemas/error",
        )
        # Send a trace activity, which will be displayed in Bot Framework Emulator
        await context.send_activity(trace_activity)


ADAPTER.on_turn_error = on_error

# Create the Bot
BOT = MyBot()


# Listen for incoming requests on /api/messages
async def messages(req: Request) -> Response:
    # Main bot message handler.
    if "application/json" in req.headers["Content-Type"]:
        body = await req.json()
    else:
        return Response(status=415)

    activity = Activity().deserialize(body)
    auth_header = req.headers["Authorization"] if "Authorization" in req.headers else ""

    response = await ADAPTER.process_activity(activity, auth_header, BOT.on_turn)
    if response:
        return json_response(data=response.body, status=response.status)
    return Response(status=201)

async def bot_service_auth(req: Request) -> Response:
    return json_response(bot_auth())

async def get_document_title(req: Request) -> Response:
    if "application/json" in req.headers["Content-Type"]:
        body = await req.json()
    else:
        return Response(status=415)
    return json_response(search.get_document_title(body["document_name"]))

async def index(request):
    return web.FileResponse('./static/index.html')

async def favicon(request):
    return web.FileResponse('./static/favicon.ico')

print("Base Path set to: %s" % (BASE_PATH))

APP = web.Application(middlewares=[aiohttp_error_middleware])
APP.router.add_post(BASE_PATH + "/api/messages", messages)
APP.router.add_get(BASE_PATH + "/api/auth", bot_service_auth)
APP.router.add_post(BASE_PATH + "/api/get_document_title", get_document_title)
APP.router.add_get(BASE_PATH + "/", index)
APP.router.add_get(BASE_PATH + "/favicon.ico", favicon)
APP.router.add_static(BASE_PATH + "/js", "static/js")

if __name__ == "__main__":
    try:
        web.run_app(APP, host="localhost", port=PORT)
    except Exception as error:
        raise error
