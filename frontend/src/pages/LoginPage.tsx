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
        <p className="eyebrow">{APP_NAME}</p>
        <h1>Sign in to safer route planning.</h1>
        <p className="muted">View nearby potholes, compare route risk, and choose a better path through Ludhiana.</p>

        <form className="auth-form" onSubmit={handleSubmit}>
          <label>
            Username
            <input value={username} onChange={(event) => setUsername(event.target.value)} autoComplete="username" required />
          </label>
          <label>
            Password
            <input type="password" value={password} onChange={(event) => setPassword(event.target.value)} autoComplete="current-password" required />
          </label>

          {error ? <ErrorBanner error={error} /> : null}
          {loading ? <LoadingPanel title="Signing in" /> : null}

          <button className="button button-primary" type="submit" disabled={loading}>
            {loading ? 'Signing in...' : 'Sign in'}
          </button>
        </form>

        <p className="auth-switch">
          New here? <Link to="/signup">Create an account</Link>
        </p>
      </section>
    </main>
  );
}
