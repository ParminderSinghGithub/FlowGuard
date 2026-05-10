type Props = {
  title: string;
  description?: string;
};

export function LoadingPanel({ title, description }: Props) {
  return (
    <div className="notice notice-loading" aria-live="polite">
      <div className="spinner" aria-hidden="true" />
      <div>
        <strong>{title}</strong>
        {description ? <p>{description}</p> : null}
      </div>
    </div>
  );
}
