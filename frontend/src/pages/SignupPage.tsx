import { FormEvent, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { APP_NAME } from '@/config';
import { signup } from '@/lib/auth';
import { ErrorBanner } from '@/components/ErrorBanner';
import { LoadingPanel } from '@/components/LoadingPanel';

export function SignupPage() {
  const navigate = useNavigate();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [passwordConfirmation, setPasswordConfirmation] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<unknown>(null);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);

    if (!username.trim() || !password || !passwordConfirmation) {
      setError('Username, password, and confirmation are required.');
      return;
    }

    if (password !== passwordConfirmation) {
      setError('Passwords do not match.');
      return;
    }

    setLoading(true);

    try {
      await signup(username.trim(), password, passwordConfirmation);
      navigate('/', { replace: true });
    } catch (caughtError) {
      setError(caughtError);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="auth-screen">
      <section className="hero-card auth-card">
        <div className="auth-brand-badge">
          <svg className="auth-brand-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
            <path d="m9 12 2 2 4-4" />
          </svg>
          <span className="eyebrow">{APP_NAME}</span>
        </div>

        <h1>Create account</h1>
        <p className="muted">Join the intelligent navigation grid with AI-predicted traffic and live road quality intelligence.</p>

        <form className="auth-form" onSubmit={handleSubmit}>
          <label>
            Username
            <input
              placeholder="Choose a username"
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
              placeholder="At least 8 characters"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              autoComplete="new-password"
              required
            />
          </label>
          <label>
            Confirm password
            <input
              type="password"
              placeholder="Confirm password"
              value={passwordConfirmation}
              onChange={(event) => setPasswordConfirmation(event.target.value)}
              autoComplete="new-password"
              required
            />
          </label>

          {error ? <ErrorBanner error={error} /> : null}
          {loading ? <LoadingPanel title="Creating account" description="Provisioning workspace credentials..." /> : null}

          <button className="button button-primary" type="submit" disabled={loading}>
            {loading ? 'Creating account...' : 'Create Account'}
          </button>
        </form>

        <p className="auth-switch">
          Already have an account? <Link to="/login">Sign in</Link>
        </p>
      </section>
    </main>
  );
}
