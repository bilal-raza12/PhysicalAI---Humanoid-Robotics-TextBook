/**
 * Root Theme Component
 *
 * This component wraps the entire Docusaurus app and adds the ChatWidget
 * for global visibility across all textbook pages.
 */

import React from 'react';
import ChatWidget from '@site/src/components/ChatWidget';

interface RootProps {
  children: React.ReactNode;
}

export default function Root({ children }: RootProps): JSX.Element {
  return (
    <>
      {children}
      <ChatWidget />
    </>
  );
}
