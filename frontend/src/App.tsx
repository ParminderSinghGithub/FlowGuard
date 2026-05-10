import type { ReactNode } from 'react';
import { Navigate, Route, Routes } from 'react-router-dom';
import { readToken } from '@/lib/auth';
import { LoginPage } from '@/pages/LoginPage';
import { SignupPage } from '@/pages/SignupPage';
import { WorkspacePage } from '@/pages/WorkspacePage';

function RequireAuth({ children }: { children: ReactNode }) {
  return readToken() ? children : <Navigate to="/login" replace />;
}

export function App() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route path="/signup" element={<SignupPage />} />
      <Route
        path="/"
        element={
          <RequireAuth>
            <WorkspacePage />
          </RequireAuth>
        }
      />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
