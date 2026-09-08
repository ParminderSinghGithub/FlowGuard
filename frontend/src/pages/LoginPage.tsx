import { FormEvent, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { APP_NAME } from '@/config';
import { login } from '@/lib/auth';
import { ErrorBanner } from '@/components/ErrorBanner';
import { LoadingPanel } from '@/components/LoadingPanel';

export function LoginPage() {
  const navigate = useNavigate();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<unknown>(null);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);

    if (!username.trim() || !password) {
      setError('Username and password are required.');
      return;
    }

    setLoading(true);

    try {
      await login(username.trim(), password);
      navigate('/', { replace: true });
    } catch (caughtError) {
      setError(caughtError);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="auth-screen">
      <section className="hero-card">
        <div className="auth-brand-badge">
          <svg className="auth-brand-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
            <path d="m9 12 2 2 4-4" />
          </svg>
          <span className="eyebrow">{APP_NAME}</span>
        </div>

        <h1>Welcome back</h1>
        <p className="muted">Access intelligent route optimization, real-time road hazard telemetry, and driving guidance.</p>

        <form className="auth-form" onSubmit={handleSubmit}>
          <label>
            Username
            <input
              placeholder="Enter your username"
              value={username}
              onChange={(event) => setUsername(event.target.value)}
              autoComplete="username"
              required
            />
          </label>
          <label>
            Password
            <input
              type="password"
              placeholder="••••••••"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              autoComplete="current-password"
              required
            />
          </label>

          {error ? <ErrorBanner error={error} /> : null}
          {loading ? <LoadingPanel title="Signing in" description="Authenticating credentials..." /> : null}

          <button className="button button-primary" type="submit" disabled={loading}>
            {loading ? 'Signing in...' : 'Sign in to Platform'}
          </button>
        </form>

        <p className="auth-switch">
          New here? <Link to="/signup">Create an account</Link>
        </p>
      </section>
    </main>
  );
}
