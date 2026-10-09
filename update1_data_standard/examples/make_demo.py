"""Create ONE invented educational example. Not SwitchLingua or a test benchmark."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import digest, write_jsonl, new_dir

out = new_dir(sys.argv[1])
text = 'Je pense que the referee is unfair.'
sid = 'demo:fr_en:001:001'
h = digest(text.encode())
write_jsonl(out/'texts.jsonl', [dict(schema_version='1.0',sequence_id=sid,
    record_id='demo:fr_en:001',item_id=1,language_pair='fr_en',languages=['fr','en'],
    source_file='INVENTED_EDUCATIONAL_EXAMPLE',source_row=1,source_path=[0],
    text=text,text_sha256=h)])
write_jsonl(out/'lid.jsonl', [dict(sequence_id=sid,text_sha256=h,
    annotation_source='assistant_illustrative_labels_not_human_gold',
    review_status='unreviewed',reviewer='',segments=[
        dict(text='Je pense que ',lang='fr'),
        dict(text='the referee is unfair',lang='en'),
        dict(text='.',lang='NEUTRAL')])])
print(out)
