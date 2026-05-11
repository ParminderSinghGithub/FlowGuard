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
        <p className="eyebrow">{APP_NAME}</p>
        <h1>Begin your intelligent journey.</h1>
        <p className="muted">Experience smart route optimization with traffic prediction and road intelligence for safer urban navigation.</p>

        <form className="auth-form" onSubmit={handleSubmit}>
          <label>
            Username
            <input value={username} onChange={(event) => setUsername(event.target.value)} autoComplete="username" required />
          </label>
          <label>
            Password
            <input type="password" value={password} onChange={(event) => setPassword(event.target.value)} autoComplete="new-password" required />
          </label>
          <label>
            Confirm password
            <input type="password" value={passwordConfirmation} onChange={(event) => setPasswordConfirmation(event.target.value)} autoComplete="new-password" required />
          </label>

          {error ? <ErrorBanner error={error} /> : null}
          {loading ? <LoadingPanel title="Creating account" /> : null}

          <button className="button button-primary" type="submit" disabled={loading}>
            {loading ? 'Creating account...' : 'Create account'}
          </button>
        </form>

        <p className="auth-switch">
          Already have an account? <Link to="/login">Sign in</Link>
        </p>
      </section>
    </main>
  );
}
