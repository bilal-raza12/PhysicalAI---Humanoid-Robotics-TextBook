/**
 * ChatWidget Component
 *
 * A ChatKit-powered chat widget for the Physical AI & Humanoid Robotics textbook.
 * Integrates with the FastAPI backend to provide grounded Q&A over textbook content.
 */

import React, { useState, useCallback } from 'react';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import styles from './styles.module.css';

// Types for our API responses
interface Citation {
  source_url: string;
  chapter: string;
  section: string;
  score: number;
  chunk_index: number;
}

interface ResponseMetadata {
  retrieval_time_ms: number;
  generation_time_ms: number;
  total_time_ms: number;
  tool_calls: string[];
}

interface QueryResponse {
  answer: string;
  citations: Citation[];
  grounded: boolean;
  refused: boolean;
  metadata: ResponseMetadata;
}

interface ErrorResponse {
  code: string;
  message: string;
  details?: Record<string, unknown>;
}

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  citations?: Citation[];
  refused?: boolean;
  timestamp: Date;
}

/**
 * ChatWidget provides a chat interface for asking questions about the textbook.
 */
export default function ChatWidget(): JSX.Element {
  const { siteConfig } = useDocusaurusContext();
  const backendUrl = (siteConfig.customFields?.backendUrl as string) || 'http://localhost:8000';

  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isOpen, setIsOpen] = useState(false);

  /**
   * Send a question to the backend API.
   */
  const sendMessage = useCallback(async () => {
    const question = input.trim();
    if (!question || isLoading) return;

    // Clear input and error immediately
    setInput('');
    setError(null);

    // Add user message
    const userMessage: Message = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: question,
      timestamp: new Date(),
    };
    setMessages(prev => [...prev, userMessage]);

    // Set loading state immediately (within 100ms per SC-002)
    setIsLoading(true);

    try {
      const response = await fetch(`${backendUrl}/api/ask`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question }),
      });

      if (!response.ok) {
        const errorData = await response.json() as { detail?: ErrorResponse };
        const errorResponse = errorData.detail;

        if (response.status === 400) {
          throw new Error(errorResponse?.message || 'Invalid request');
        } else if (response.status === 500) {
          throw new Error(errorResponse?.message || 'Server error. Please try again later.');
        } else {
          throw new Error('An unexpected error occurred');
        }
      }

      const data = await response.json() as QueryResponse;

      // Add assistant message with citations
      const assistantMessage: Message = {
        id: `assistant-${Date.now()}`,
        role: 'assistant',
        content: data.answer,
        citations: data.citations,
        refused: data.refused,
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, assistantMessage]);

    } catch (err) {
      // Handle network errors
      if (err instanceof TypeError && err.message.includes('fetch')) {
        setError('Unable to connect to the server. Please try again later.');
      } else if (err instanceof Error) {
        setError(err.message);
      } else {
        setError('An unexpected error occurred');
      }
    } finally {
      setIsLoading(false);
    }
  }, [input, isLoading, backendUrl]);

  /**
   * Handle keyboard submission.
   */
  const handleKeyDown = useCallback((e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }, [sendMessage]);

  /**
   * Render a citation as a clickable link.
   */
  const renderCitation = (citation: Citation, index: number) => (
    <a
      key={index}
      href={citation.source_url}
      target="_blank"
      rel="noopener noreferrer"
      className={styles.citationLink}
      title={`${citation.chapter} - ${citation.section} (Score: ${citation.score.toFixed(2)})`}
    >
      [{index + 1}]
    </a>
  );

  /**
   * Render a message with optional citations.
   */
  const renderMessage = (message: Message) => (
    <div
      key={message.id}
      className={`${styles.message} ${styles[message.role]} ${message.refused ? styles.refused : ''}`}
    >
      <div className={styles.messageContent}>
        {message.content}
      </div>
      {message.citations && message.citations.length > 0 && (
        <div className={styles.citations}>
          <span className={styles.citationsLabel}>Sources: </span>
          {message.citations.map((c, i) => renderCitation(c, i))}
        </div>
      )}
      {message.refused && (
        <div className={styles.refusalHint}>
          Try asking about topics covered in the Physical AI & Humanoid Robotics textbook.
        </div>
      )}
    </div>
  );

  return (
    <>
      {/* Toggle button */}
      <button
        className={styles.toggleButton}
        onClick={() => setIsOpen(!isOpen)}
        aria-label={isOpen ? 'Close chat' : 'Open chat'}
      >
        {isOpen ? '×' : '💬'}
      </button>

      {/* Chat widget */}
      {isOpen && (
        <div className={styles.widget}>
          <div className={styles.header}>
            <span>Ask the Textbook</span>
            <button
              className={styles.closeButton}
              onClick={() => setIsOpen(false)}
              aria-label="Close chat"
            >
              ×
            </button>
          </div>

          <div className={styles.messagesContainer}>
            {messages.length === 0 && (
              <div className={styles.welcomeMessage}>
                Ask any question about Physical AI & Humanoid Robotics!
              </div>
            )}
            {messages.map(renderMessage)}
            {isLoading && (
              <div className={`${styles.message} ${styles.assistant}`}>
                <div className={styles.typingIndicator}>
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            )}
          </div>

          {error && (
            <div className={styles.errorMessage}>
              {error}
            </div>
          )}

          <div className={styles.inputContainer}>
            <textarea
              className={styles.input}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask a question..."
              disabled={isLoading}
              rows={1}
            />
            <button
              className={styles.sendButton}
              onClick={sendMessage}
              disabled={isLoading || !input.trim()}
              aria-label="Send message"
            >
              {isLoading ? '...' : '→'}
            </button>
          </div>
        </div>
      )}
    </>
  );
}
