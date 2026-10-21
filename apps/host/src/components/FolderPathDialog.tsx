import { useEffect, useState } from 'react';
import { tr } from '../lib/i18n';
import { useFolderPrompt } from '../lib/folder-pick';
import { api } from '../lib/client';
import type { LocalModelFolder } from '../lib/types';
import { Btn, Dialog, Icon } from './ui';

/** The fallback for a system without a folder chooser: say so in plain words and take the folder's path as text or browse folders. */
export function FolderPathDialog() {
  const prompt = useFolderPrompt();
  const [path, setPath] = useState('');
  const [folder, setFolder] = useState<LocalModelFolder | null>(null);
  const [loading, setLoading] = useState(false);

  async function browse(dir?: string) {
    setLoading(true);
    try {
      const url = dir ? `/api/local-models/browse?directory=${encodeURIComponent(dir)}` : '/api/local-models/browse';
      const res = await api.get<LocalModelFolder>(url);
      setFolder(res);
      setPath(res.directory);
    } catch {
      /* keep current typed path */
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (prompt) {
      setPath('');
      void browse();
    }
  }, [prompt]);

  const submit = () => { const text = path.trim(); if (text) prompt?.resolve(text); };
  return <Dialog open={!!prompt} compact title={tr("Choose a folder")} onClose={() => prompt?.resolve(null)}
    actions={<><Btn onClick={() => prompt?.resolve(null)}>{tr("Cancel")}</Btn><Btn variant="primary" disabled={!path.trim()} onClick={submit}>{tr("Use this folder")}</Btn></>}>
    <form className="stack" onSubmit={event => { event.preventDefault(); submit(); }}>
      <p className="small" role="status">{tr(prompt?.message ?? '')}</p>
      <div className="row" style={{ gap: 8, alignItems: 'flex-end' }}>
        <label className="input-field" style={{ flex: 1 }}><span>{tr("Folder path")}</span>
          <input className="st-input" autoFocus value={path} onChange={event => setPath(event.target.value)} placeholder={tr("D:\\Projects\\my-app")} spellCheck={false} aria-label={tr("Folder path")}/>
        </label>
        <Btn type="button" size="sm" disabled={loading || !path.trim()} onClick={() => void browse(path.trim())}>{tr("Browse")}</Btn>
      </div>
      {folder ? (
        <div className="onboard-folder-browser" style={{ marginTop: 8 }}>
          <div className="onboard-folder-shortcuts" style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 8 }}>
            {folder.shortcuts?.map(item => <Btn key={item.path} type="button" size="sm" disabled={loading} onClick={() => void browse(item.path)}>{tr(item.name)}</Btn>)}
            {folder.parent ? <Btn type="button" size="sm" disabled={loading} onClick={() => void browse(folder.parent!)}>↑ {tr("Up a folder")}</Btn> : null}
          </div>
          <div className="onboard-folder-files" style={{ maxHeight: 220, overflowY: 'auto', border: '1px solid var(--st-line)', borderRadius: 10, padding: 6 }}>
            {folder.entries.filter(e => e.kind === 'folder').slice(0, 60).map(e => (
              <button key={e.path} type="button" style={{ display: 'flex', width: '100%', alignItems: 'center', gap: 8, padding: '6px 8px', textAlign: 'left', background: 'transparent', border: 'none', cursor: 'pointer', borderRadius: 6 }} onClick={() => void browse(e.path)}>
                <Icon name="folder" size={16}/><span style={{ flex: 1 }}>{e.name}</span><Icon name="chevron" size={12}/>
              </button>
            ))}
            {folder.entries.filter(e => e.kind === 'folder').length === 0 ? <p className="small muted" style={{ padding: 6 }}>{tr("No subfolders in this folder.")}</p> : null}
          </div>
        </div>
      ) : null}
    </form>
  </Dialog>;
}
