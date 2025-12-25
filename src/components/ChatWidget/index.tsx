/**
 * ChatWidget Component - Modern UI
 *
 * A beautifully designed chat widget for the Physical AI & Humanoid Robotics textbook.
 * Features smooth animations, avatars, suggestion chips, and excellent UX.
 */

import React, { useState, useCallback, useRef, useEffect } from 'react';
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

// Suggestion chips for quick questions
const SUGGESTIONS = [
  'What is ROS 2?',
  'Explain digital twins',
  'How does NVIDIA Isaac work?',
];

/**
 * Format time to readable string
 */
function formatTime(date: Date): string {
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

/**
 * ChatWidget provides a modern chat interface for asking questions about the textbook.
 */
export default function ChatWidget(): JSX.Element {
  const { siteConfig } = useDocusaurusContext();
  const backendUrl = (siteConfig.customFields?.backendUrl as string) || 'http://localhost:8000';

  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isOpen, setIsOpen] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  // Focus input when widget opens
  useEffect(() => {
    if (isOpen) {
      setTimeout(() => inputRef.current?.focus(), 100);
    }
  }, [isOpen]);

  /**
   * Send a question to the backend API.
   */
  const sendMessage = useCallback(async (questionText?: string) => {
    const question = (questionText || input).trim();
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

    // Set loading state immediately
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
        setError('Unable to connect to the server. Please check if the backend is running.');
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
   * Clear chat history
   */
  const clearChat = useCallback(() => {
    setMessages([]);
    setError(null);
  }, []);

  /**
   * Handle suggestion chip click
   */
  const handleSuggestionClick = useCallback((suggestion: string) => {
    sendMessage(suggestion);
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
      title={`${citation.chapter} - ${citation.section}`}
    >
      <span className={styles.citationIcon}>📄</span>
      {citation.chapter}
    </a>
  );

  /**
   * Render a message with avatar and optional citations.
   */
  const renderMessage = (message: Message) => (
    <div
      key={message.id}
      className={`${styles.messageWrapper} ${styles[message.role]} ${message.refused ? styles.refused : ''}`}
    >
      <div className={`${styles.avatar} ${styles[message.role]}`}>
        {message.role === 'user' ? '👤' : '🤖'}
      </div>
      <div className={styles.messageBubble}>
        <div className={styles.messageContent}>
          {message.content}
        </div>
        {message.citations && message.citations.length > 0 && (
          <div className={styles.citations}>
            <span className={styles.citationsLabel}>📚 Sources</span>
            {message.citations.slice(0, 3).map((c, i) => renderCitation(c, i))}
          </div>
        )}
        {message.refused && (
          <div className={styles.refusalHint}>
            <span className={styles.refusalIcon}>💡</span>
            Try asking about topics covered in the textbook.
          </div>
        )}
        <div className={styles.messageTime}>
          {formatTime(message.timestamp)}
        </div>
      </div>
    </div>
  );

  return (
    <>
      {/* Toggle button */}
      <button
        className={`${styles.toggleButton} ${isOpen ? styles.open : ''}`}
        onClick={() => setIsOpen(!isOpen)}
        aria-label={isOpen ? 'Close chat' : 'Open chat'}
      >
        <span className={styles.toggleIcon}>
          {isOpen ? '✕' : '💬'}
        </span>
      </button>

      {/* Chat widget */}
      {isOpen && (
        <div className={styles.widget}>
          {/* Header */}
          <div className={styles.header}>
            <div className={styles.headerIcon}>🤖</div>
            <div className={styles.headerInfo}>
              <h3 className={styles.headerTitle}>AI Assistant</h3>
              <p className={styles.headerSubtitle}>Ask anything about the textbook</p>
            </div>
            <div className={styles.headerActions}>
              <button
                className={styles.headerButton}
                onClick={clearChat}
                aria-label="Clear chat"
                title="Clear chat"
              >
                🗑️
              </button>
              <button
                className={styles.closeButton}
                onClick={() => setIsOpen(false)}
                aria-label="Close chat"
              >
                ✕
              </button>
            </div>
          </div>

          {/* Messages */}
          <div className={styles.messagesContainer}>
            {messages.length === 0 ? (
              <div className={styles.welcomeContainer}>
                <div className={styles.welcomeIcon}>📚</div>
                <h4 className={styles.welcomeTitle}>Welcome!</h4>
                <p className={styles.welcomeText}>
                  I'm your AI assistant for the Physical AI & Humanoid Robotics textbook.
                  Ask me anything about ROS 2, digital twins, NVIDIA Isaac, or VLA!
                </p>
                <div className={styles.suggestions}>
                  {SUGGESTIONS.map((suggestion, index) => (
                    <button
                      key={index}
                      className={styles.suggestionChip}
                      onClick={() => handleSuggestionClick(suggestion)}
                    >
                      {suggestion}
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              <>
                {messages.map(renderMessage)}
                {isLoading && (
                  <div className={styles.typingWrapper}>
                    <div className={`${styles.avatar} ${styles.assistant}`}>🤖</div>
                    <div className={styles.typingIndicator}>
                      <span></span>
                      <span></span>
                      <span></span>
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </>
            )}
          </div>

          {/* Error message */}
          {error && (
            <div className={styles.errorMessage}>
              <span className={styles.errorIcon}>⚠️</span>
              <span>{error}</span>
              <button
                className={styles.errorDismiss}
                onClick={() => setError(null)}
                aria-label="Dismiss error"
              >
                ✕
              </button>
            </div>
          )}

          {/* Input */}
          <div className={styles.inputContainer}>
            <div className={styles.inputWrapper}>
              <textarea
                ref={inputRef}
                className={styles.input}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask a question..."
                disabled={isLoading}
                rows={1}
                maxLength={500}
              />
              {input.length > 400 && (
                <span className={`${styles.charCount} ${input.length > 480 ? styles.error : styles.warning}`}>
                  {input.length}/500
                </span>
              )}
            </div>
            <button
              className={styles.sendButton}
              onClick={() => sendMessage()}
              disabled={isLoading || !input.trim()}
              aria-label="Send message"
            >
              <span className={styles.sendIcon}>➤</span>
            </button>
          </div>
        </div>
      )}
    </>
  );
}
