import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout'
import Upload from './pages/Upload'
import Mapping from './pages/Mapping'
import Processing from './pages/Processing'
import Dashboard from './pages/Dashboard'
import LoanDetail from './pages/LoanDetail'
import HowItWorks from './pages/HowItWorks'

export default function App() {
  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<Upload />} />
          <Route path="/mapping/:uploadId" element={<Mapping />} />
          <Route path="/processing/:jobId" element={<Processing />} />
          <Route path="/dashboard/:jobId" element={<Dashboard />} />
          <Route path="/loan/:jobId/:loanId" element={<LoanDetail />} />
          <Route path="/how-it-works" element={<HowItWorks />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  )
}
