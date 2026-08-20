import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { ResultGrid } from '../src/components/Grid/ResultGrid';
import { useSearchStore } from '../src/stores/useSearchStore';

const initialState = useSearchStore.getInitialState();

describe('ResultGrid critical states', () => {
  beforeEach(() => {
    useSearchStore.setState(initialState, true);
  });

  afterEach(() => {
    cleanup();
  });

  it('renders a critical search error', () => {
    useSearchStore.setState({
      status: 'error',
      error: 'Unable to reach the search backend.',
      results: [],
    });

    const message = screen.queryByText('Unable to reach the search backend.');
    expect(message).toBeNull();

    render(<ResultGrid />);

    expect(screen.getByText('Unable to reach the search backend.').className).toContain(
      'error',
    );
  });

  it('renders the successful empty-result state', () => {
    useSearchStore.setState({
      status: 'success-empty',
      error: null,
      results: [],
    });

    render(<ResultGrid />);

    expect(screen.getByText('No matching keyframes.').textContent).toBe(
      'No matching keyframes.',
    );
  });

  it('hides the development fixture control in default mode', () => {
    render(<ResultGrid />);

    expect(
      screen.queryByRole('button', { name: 'Fixture Search (dev)' }),
    ).toBeNull();
  });
});
