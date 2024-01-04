from whoosh import fields, index, analysis, qparser, query
from whoosh.analysis import CharsetFilter
from whoosh.support.charset import accent_map
import os

# Code from Brian McFee
def create_index_writer(index_path):
    ''' Constructs a whoosh index writer, which has ID, artist and title fields'''
    if not os.path.exists(index_path):
        os.makedirs(index_path, exist_ok=True)
    
    A = (analysis.StandardAnalyzer(stoplist=None, minsize=1) | CharsetFilter(accent_map))
    
    Schema = fields.Schema(
        id=fields.ID(stored=True),
        path=fields.TEXT(stored=True),
        artist=fields.TEXT(stored=True, analyzer=A),
        title=fields.TEXT(stored=True, analyzer=A)
    )
    
    index_instance = index.create_in(index_path, Schema)
    return index_instance.writer()

# Code from Brian McFee
def create_index(index_path, track_list):
    ''' Creates a whoosh index directory for the MSD'''
    writer = create_index_writer(index_path)
    for entry in track_list:
        writer.add_document(**entry)
    writer.commit()

def get_whoosh_index(index_path):
    ''' Get a whoosh searcher object from a whoosh index path'''
    return index.open_dir(index_path)

def search(searcher, schema, artist, title, threshold=20):
    ''' Search for an artist - title pair and return the best match'''

    artist = str(artist)
    title = str(title)
    
    arparser = qparser.QueryParser('artist', schema)
    tiparser = qparser.QueryParser('title', schema)
    
    q = query.And([arparser.parse(artist), tiparser.parse(title)])
    
    results = searcher.search(q)
    
    if len(results) > 0:
        return [[r['id'], r['artist'], r['title']] for r in results if r.score > threshold]
    else:
        return []

