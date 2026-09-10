"""Import verified prepared data or the existing teacher500 reference into a NEW working root/run."""
import argparse
from pathlib import Path
import shutil
import json

import course
from common import digest_file, read_json, read_jsonl, write_json


def import_prepared(root, source):
    manifest = read_json(source / 'manifest.json')
    for name, sha in manifest['files'].items():
        target = (source / name).resolve()
        course.need(target.is_relative_to(source.resolve()), 'Unsafe material path')
        course.need(digest_file(target) == sha, 'Prepared material hash mismatch: ' + name)
    for name, key in [('candidates', 'candidate_count'), ('dev', 'dev_count'), ('test', 'test_count')]:
        course.need(len(read_jsonl(source / (name + '.jsonl'))) == course.CFG[key], 'Material count differs from course config')
    for source_key, cfg_id, cfg_rev in [('numina', 'numina_id', 'numina_revision'), ('gsm8k', 'gsm8k_id', 'gsm8k_revision')]:
        course.need(manifest['source'][source_key] == [course.CFG[cfg_id], course.CFG[cfg_rev]], 'Material revision mismatch')
    course.need(not (root / 'data').exists(), 'Working data already exists; do not overwrite')
    (root / 'data').mkdir(parents=True)
    for name in manifest['files']:
        shutil.copy2(source / name, root / 'data' / name)
    write_json(root / 'data/manifest.json', {'time': course.now(), 'source': manifest['source'],
        'mode': 'verified_prepared_materials', 'material_manifest_sha256': digest_file(source / 'manifest.json'),
        'config_sha256': digest_file(course.CODE / 'configs/course.json'), 'files': manifest['files']})


def import_teacher(root, source, run):
    course.data_ready(root)
    manifest = read_json(source / 'manifest.json')
    for name, sha in manifest['files'].items():
        target = (source / name).resolve()
        course.need(target.is_relative_to(source.resolve()), 'Unsafe material path')
        course.need(digest_file(target) == sha, 'Teacher material hash mismatch: ' + name)
    for name, sha in manifest['evaluation_files'].items():
        course.need(digest_file(root / 'data' / name) == sha, 'Reference teacher500 requires its original dev/test split')
    rows = read_json(source / 'train.json')
    course.need(len(rows) == 500 and len({r['source_id'] for r in rows}) == 500, 'Expected 500 distinct teacher examples')
    course.need(all(r.get('system') == course.CFG['system'] for r in rows), 'Teacher data system prompt differs')
    folder = course.run_path(root, run)
    course.need(not folder.exists(), 'Run exists; preserve it')
    (folder / 'dataset').mkdir(parents=True)
    for name in ['train.json', 'dataset_info.json']:
        shutil.copy2(source / name, folder / 'dataset' / name)
    shutil.copy2(source / 'generation_audit.jsonl', folder / 'generation_audit.jsonl')
    write_json(folder / 'request.json', {'time': course.now(), 'mode': 'pilot500', 'target': 500, 'config': course.CFG,
        'origin': 'existing_teacher500_reference_not_generated_in_this_run'})
    write_json(folder / 'generation_complete.json', {'time': course.now(), 'mode': 'pilot500', 'accepted': 500,
        'origin': 'existing_teacher500_reference_not_generated_in_this_run',
        'source_manifest_sha256': digest_file(source / 'manifest.json'),
        'train_sha256': digest_file(folder / 'dataset/train.json'),
        'data_manifest_sha256': digest_file(root / 'data/manifest.json')})


def main():
    import os
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('kind', choices=['prepared', 'teacher500'])
    p.add_argument('--root', type=Path, default=Path(os.environ.get('COURSE_DATA_ROOT', '/root/distill-work')))
    p.add_argument('--run', default='instruct500-reference-01')
    args = p.parse_args()
    if args.kind == 'prepared':
        import_prepared(args.root, course.CODE / 'materials/prepared')
    else:
        import_teacher(args.root, course.CODE / 'materials/teacher500', args.run)
    print('Imported verified materials:', args.kind, '; no model training/generation was performed.')


if __name__ == '__main__':
    main()
