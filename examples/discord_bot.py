# pip install discord
# pip uninstall discord.py
# pip install py-cord
import os
import time

# from typing import Optional
import discord
import dotenv
import requests
from huggingface_hub import HfApi


api = HfApi()
dotenv.load_dotenv()
bot = discord.Bot()

sources = {
    "darksun": "https://darksun.fandom.com/wiki",
    "dnd4e": "https://dnd4.fandom.com/wiki",
    "dnd5e": "https://dnd-5e.fandom.com/wiki",
    "dragonlance": "https://dragonlance.fandom.com/wiki",
    "eberron": "https://eberron.fandom.com/wiki",
    "exandria": "https://criticalrole.fandom.com/wiki",
    "forgottenrealms": "https://forgottenrealms.fandom.com/wiki",
    "greyhawk": "https://greyhawkonline.com/greyhawkwiki",
    "planescape": "https://planescape.fandom.com/wiki",
    "ravenloft": "https://www.fraternityofshadows.com/wiki",
    "spelljammer": "https://spelljammer.fandom.com/wiki",
}


@bot.event
async def on_ready():
    "On bot startup."
    print(f"Logged in as {bot.user.name}")


def check_server(response: requests.models.Response, clock: int):
    """Check Huggingface API for space status.

    Args:
        response (requests.models.Response): before space query
        clock (int): number of 30 second intervals before giving up

    Returns:
        requests.models.Response: after space query
    """
    while response.json()["runtime"]["stage"] != "RUNNING" or clock != 0:
        response = requests.get(
            "https://huggingface.co/api/spaces/TotalSundae/dungeons-and-dragons",
            params={},
            headers={"Authorization": f'Bearer {str(os.getenv("HF_TOKEN"))}'},
            timeout=60,
        )
        print(response.json()["runtime"]["stage"])
        time.sleep(30)
        clock -= 1
    return response


@bot.command(description="Ask Volo a question.")
async def ask(ctx, prompt: str):
    """Question asked by user via slash command.
    Args:
        ctx (Context): discord command context metadata
        prompt (str): command body
    """
    # API Endpoint for the POST request
    api_url = "https://totalsundae-dungeons-and-dragons.hf.space"
    # Your API Request Payload
    payload = {
        "prompt": prompt,
    }
    response = requests.get(
        "https://huggingface.co/api/spaces/TotalSundae/dungeons-and-dragons",
        params={},
        headers={"Authorization": f'Bearer {str(os.getenv("HF_TOKEN"))}'},
        timeout=60,
    )
    print(response.json()["runtime"]["stage"])
    while response.json()["runtime"]["stage"] != "RUNNING":
        match response.json()["runtime"]["stage"]:
            case "BUILD_ERROR":
                await ctx.respond(
                    content="Server Error, please notify an Administrator."
                )
                break
            case "BUILDING":
                await ctx.respond(content="Server currently starting, please wait.")
                response = check_server(response, 10)
            case "PAUSED":
                await ctx.respond(
                    content="Server Paused, please try again at a later time."
                )
                break
            case "RUNNING":
                await ctx.respond(f"{ctx.author.mention} asked: {prompt}")
                response = requests.post(
                    f"{api_url}/query",
                    headers={"Connection": "close"},
                    json=payload,
                    timeout=3600,
                )
                response = response.json()
                embed = discord.Embed(
                    title=response["question"],
                    description=response["answer"],
                    color=0xB431BD,
                )
                for source in response["source_documents"]:
                    page, wiki = source["metadata"]["source"].split(" - ")
                    embed.add_field(
                        name=f'{sources[wiki]}/{page.replace(" ", "_")}',
                        # First 400 characters of source
                        value=f'{source["page_content"][0:200]}...',
                        inline=False,
                    )
                # Printing the API response to the user
                await ctx.respond(content=ctx.author.mention, embed=embed)
            case "SLEEPING":
                await ctx.respond(content="Server Offline, Restarting, please wait.")
                api.restart_space(
                    repo_id="TotalSundae/dungeons-and-dragons",
                    token=str(os.getenv("HF_TOKEN")),
                )
                response = check_server(response, 5)


bot.run(str(os.getenv("TOKEN")))
