# pip install discord py-cord
import os

# from typing import Optional
import discord
import dotenv
from requests import post

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
async def ask(
    ctx,
    prompt: str,
    # num_sources: Optional[int],
    # temperature: Optional[float],
    # repeat_penalty: Optional[float],
    # top_k: Optional[int],
    # top_p: Optional[float],
):
    """Question asked by user via slash command.

    Args:
        ctx (Context): discord command context metadata
        prompt (str): command body
    """
    # API Endpoint for the POST request
    api_url = "http://localhost:8000/query"
    # Your API Request Payload
    payload = {
        "prompt": prompt,
        # "num_sources": num_sources,
        # "temperature": temperature,
        # "repeat_penalty": repeat_penalty,
        # "top_k": top_k,
        # "top_p": top_p,
    }
    # await ctx.respond(f"{ctx.author.mention} asked: _{prompt}_")
    # Making the POST Request

    await ctx.respond(f"{ctx.author.mention} asked: {prompt}")

    response = post(api_url, json=payload, timeout=3600).json()

    embed = discord.Embed(title=response["answer"], description="", color=0xB431BD)
    for source in response["source_documents"]:
        page, wiki = source["metadata"]["source"].split(" - ")
        embed.add_field(
            name=f'{sources[wiki]}/{page.replace(" ", "_")}',
            # First 400 characters of source
            value=f'{source["page_content"][0:200]}...',
            inline=False,
        )
    # Printing the API response to the user
    await ctx.respond(content=response["question"], embed=embed)


# Run the bot with your token
bot.run(str(os.getenv("TOKEN")))
