# pip install discord
# pip uninstall discord.py
# pip install py-cord
import os
import time

# from typing import Optional
import discord
import dotenv
from requests import get, post


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


@bot.command(description="Ask Volo a question.")
async def ask(ctx, prompt: str):
    """Question asked by user via slash command.
    Args:
        ctx (Context): discord command context metadata
        prompt (str): command body
    """
    # API Endpoint for the POST request
    api_url = "http://localhost:8000"
    # Your API Request Payload
    payload = {
        "prompt": prompt,
    }

    if not get(f"{api_url}/ping", timeout=5).json() == {"status": "Healthy"}:
        await ctx.respond(
            content="Looks like I'm offline at the moment, I've asked the kind gods at Huggingface to restart my engine and I'll get back to you in a jiffy!"
        )
        time.sleep(240)
    await ctx.respond(f"{ctx.author.mention} asked: {prompt}")
    response = post(f"{api_url}/query", json=payload, timeout=3600)
    response = response.json()
    embed = discord.Embed(
        title=response["question"], description=response["answer"], color=0xB431BD
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


bot.run(str(os.getenv("TOKEN")))
