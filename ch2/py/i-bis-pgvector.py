from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres.vectorstores import PGVector
from langchain_core.documents import Document
import uuid


# See docker command above to launch a postgres instance with pgvector enabled.
connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
embeddings_model = OpenAIEmbeddings()
collection_name = "langchain"

vector_store = PGVector(
    embeddings=embeddings_model,
    collection_name=collection_name,
    connection=connection
)

# The store is now ready to query
results = vector_store.similarity_search("Greece", k=4)
print(results)


"""
    
C:\Users\tomasz.ziss\Documents\learning-langchain\ch2\py\i-bis-pgvector.py:1: DeprecationWarning: `langchain-community` 
is being sunset and is no longer actively maintained. See https://github.com/langchain-ai/langchain-community/issues/674 for details 
and migration guidance toward standalone integration packages.

from langchain_community.document_loaders import TextLoader
[Document(id='a40a7815-7aeb-4687-be29-974fa46e084b', metadata={'source': './test.txt'}, page_content='We may, perhaps, find some suggestion of this combined distinctness and\ncomprehensiveness in the aspect and configurationof Greece itself;\nin its manifold varieties of soil, and climate, and scenery, and\nproductions; in the exquisite clearness with which the features of its\nlandscape are defined; and the admirable development of coast-line by\nwhich all parts of its territory, while preserving their political\nindependence, were brought into safe and speedy communication with\none another. The industrial and commercial habits of the people,\nnecessitating a well-marked division of labour and a regulated\ndistribution of commodities, gave a further impulse in the same\ndirection.'), Document(id='a9effd0b-e2ab-493d-93f1-297e71195a30', metadata={'source': './test.txt'}, page_content="Maritime Trade and Colonization\nGreece's geographical location fostered a strong maritime tradition, with numerous islands and a rugged coastline facilitating trade across the Mediterranean and Black Seas. Greek traders established colonies in regions such as Sicily, Asia Minor, and North Africa, spreading Greek culture and goods while benefiting from the exchange of resources and ideas.\nArt, Architecture, and Engineering\nThe Greeks were exceptional innovators, leaving an indelible mark on art, architecture, and engineering. Their contributions continue to influence modern aesthetics and structural design.\nSculpture and Pottery"), Document(id='3abaeefb-b169-477a-b2fc-8acb2d266ecc', metadata={'source': './test.txt'}, page_content='The Polis: Heart of Greek Society\nAt the core of ancient Greek life was the polis, a city-state that served as the primary political and social unit. Each polis was an independent entity with its own government, laws, and customs, fostering a strong sense of community and civic responsibility among its citizens. The most famous of these city-states—Athens, Sparta, Corinth, and Thebes—each had distinct characteristics and governance structures, contributing to a diverse and dynamic Greek landscape.\nAthens: Cradle of Democracy\nAthens is renowned as the birthplace of democracy, where citizens participated directly in decision-making processes. Assemblies were held regularly, allowing male citizens to voice their opinions and vote on legislation, military actions, and other crucial matters. This participatory governance encouraged active civic engagement and laid the foundation for modern democratic systems.\nSparta: Military Prowess and Discipline'), Document(id='d1b0c673-70c1-4872-95b2-d691ce371dd7', metadata={'source': './test.txt'}, page_content='During the two centuries that ended with the close of the Peloponnesian\nwar, a single race, weak numerically, and weakened still further\nby political disunion, simultaneously developed all the highest\nhuman faculties to an extent possibly rivalled but certainly not\nsurpassed by the collective efforts of that vastly greater population\nwhich now wields the accumulated resources of modern Europe. This\nrace, while maintaining a precarious foothold on the shores of the\nMediterranean by repeated prodigies of courage and genius, contributed\na new element to civilisation which has been the mainspring of all\nsubsequent progress, but which, as it expanded into wider circles and\nencountered an increasing resistance from without, unavoidably lost\nsome of the enormous elasticity that characterised its earliest and\nmost concentrated reaction. It was the just boast of the Greek that\nto Asiatic refinement and Thracian valour he joined a disinterested')]
    
"""