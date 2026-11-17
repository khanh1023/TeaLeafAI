import { NavLink, Routes, Route, Navigate, Link, useLocation } from 'react-router-dom';
import { Leaf, House, ScanSearch, BookOpen, History, CircleHelp, Settings, Menu, X, ArrowUpRight } from 'lucide-react';
import { useEffect, useState } from 'react';
import Home from './pages/Home';
import Diagnosis from './pages/Diagnosis';
import Library from './pages/Library';
import HistoryPage from './pages/History';
import Guide from './pages/Guide';
import Admin from './pages/Admin';
const nav = [['/', 'Trang chủ', House, true], ['/chan-doan', 'Phân tích', ScanSearch], ['/thu-vien', 'Thư viện', BookOpen], ['/lich-su', 'Lịch sử', History], ['/huong-dan', 'Hướng dẫn', CircleHelp], ['/quan-tri', 'Hệ thống', Settings]];

export default function App() {
  const [open, setOpen] = useState(false);
  const location = useLocation();
  useEffect(() => { setOpen(false); window.scrollTo({ top: 0, behavior: 'instant' }); }, [location.pathname]);
  return <div className="shell"><a className="skip-link" href="#main-content">Đến nội dung chính</a><header className="header"><div className="header-inner"><Link to="/" className="brand" aria-label="Tea Leaf AI — Trang chủ"><span className="brand-icon"><Leaf size={25}/></span><span><b>Tea Leaf<span> AI</span></b><small>HIỂU LÁ · CHĂM VƯỜN</small></span></Link><button className="menu" aria-label={open ? 'Đóng menu' : 'Mở menu'} aria-expanded={open} aria-controls="main-navigation" onClick={() => setOpen(!open)}>{open ? <X/> : <Menu/>}</button><nav id="main-navigation" className={open ? 'nav show' : 'nav'} aria-label="Điều hướng chính">{nav.map(([to, label, Icon, end]) => <NavLink key={to} to={to} end={end}><Icon size={16}/>{label}</NavLink>)}</nav><Link className="nav-cta" to="/chan-doan">Thử ngay <ArrowUpRight size={16}/></Link></div></header><main id="main-content"><Routes><Route path="/" element={<Home/>}/><Route path="/chan-doan" element={<Diagnosis/>}/><Route path="/thu-vien" element={<Library/>}/><Route path="/lich-su" element={<HistoryPage/>}/><Route path="/huong-dan" element={<Guide/>}/><Route path="/quan-tri" element={<Admin/>}/><Route path="*" element={<Navigate to="/" replace/>}/></Routes></main><footer className="footer"><div className="footer-inner"><div><Link to="/" className="footer-brand"><Leaf size={23}/> Tea Leaf AI</Link><p>Một góc nhìn từ AI cho chiếc lá trong vườn bạn.</p><small>Công cụ tham khảo · Không thay thế chuyên gia nông nghiệp</small></div><div className="footer-links"><Link to="/huong-dan">Hướng dẫn sử dụng</Link><a href="https://github.com/khanh1023/TeaLeafAI" target="_blank" rel="noreferrer">Mã nguồn mở <ArrowUpRight size={14}/></a><span>Được xây dựng với công cụ miễn phí.</span></div></div><div className="footer-bottom"><span>Tea Leaf AI · 8 nhóm phân loại lá chè</span><span>Quan sát kỹ. Chăm sóc có cơ sở.</span></div></footer></div>;
}
