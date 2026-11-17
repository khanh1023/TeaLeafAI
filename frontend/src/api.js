const BASE=(import.meta.env.VITE_API_URL||"http://127.0.0.1:8000").replace(/\/$/,"");
export async function health(){const r=await fetch(`${BASE}/api/health`);if(!r.ok)throw new Error("Không kết nối được máy chủ AI");return r.json();}
export async function diagnose(file){const f=new FormData();f.append("file",file);const r=await fetch(`${BASE}/api/diagnosis`,{method:"POST",body:f});const d=await r.json().catch(()=>({}));if(!r.ok)throw new Error(d.detail||"Không thể phân tích ảnh");return d;}
export {BASE};