from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

# Sample Python code to split.
python_code = '''
class Player:
    def __init__(self, name, runs):
        self.name = name
        self.runs = runs

    def add_runs(self, runs):
        self.runs += runs


class Team:
    def __init__(self, name):
        self.name = name
        self.players = []

    def add_player(self, player):
        self.players.append(player)

    def total_runs(self):
        return sum(p.runs for p in self.players)


def print_scorecard(team):
    for player in team.players:
        print(player.name, player.runs)
'''

# Uses Python-specific separators: tries to split between classes first,
# then between functions, then falls back to paragraphs, lines and words.
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=300,    # max characters per chunk
    chunk_overlap=0,   # characters shared between neighbouring chunks
)

chunks = splitter.split_text(python_code)

print(len(chunks), "chunks")

for chunk in chunks:
    print(chunk)
    print("-" * 50)
