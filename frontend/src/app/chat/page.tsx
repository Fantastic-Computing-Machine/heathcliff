"use client";

import { useRef, useState } from "react";
import { AppShell } from "@/components/app-shell";
import {
  Conversation,
  ConversationContent,
} from "@/components/ai-elements/conversation";
import {
  Message,
  MessageContent,
  MessageResponse,
} from "@/components/ai-elements/message";
import {
  PromptInput,
  PromptInputBody,
  PromptInputButton,
  PromptInputFooter,
  PromptInputSubmit,
  PromptInputTextarea,
  PromptInputTools,
} from "@/components/ai-elements/prompt-input";
import { PlusIcon } from "lucide-react";

type ChatMessage = { id: string; from: "user" | "assistant"; text: string };
type RuntimeEvent = {
  kind: string;
  payload: { text?: string; response?: string; error?: string };
};

function readEvents(chunk: string): RuntimeEvent[] {
  return chunk.split("\n\n").flatMap((event) => {
    const data = event
      .split("\n")
      .find((line) => line.startsWith("data: "))
      ?.slice(6);
    if (!data) return [];
    try {
      return [JSON.parse(data) as RuntimeEvent];
    } catch {
      return [];
    }
  });
}

export default function ChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const threadId = useRef<string | undefined>(undefined);

  function startNewChat() {
    threadId.current = undefined;
    setError(null);
    setMessages([]);
  }

  function updateAssistant(id: string, update: (text: string) => string) {
    setMessages((current) =>
      current.map((message) =>
        message.id === id ? { ...message, text: update(message.text) } : message
      )
    );
  }

  async function submit(content: string) {
    const text = content.trim();
    if (!text || isStreaming) return;

    const assistantId = crypto.randomUUID();
    setError(null);
    setIsStreaming(true);
    setMessages((current) => [
      ...current,
      { id: crypto.randomUUID(), from: "user", text },
      { id: assistantId, from: "assistant", text: "" },
    ]);

    try {
      const started = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content: text, threadId: threadId.current }),
      });
      const response = (await started.json()) as {
        threadId?: string;
        cursor?: number;
        error?: string;
      };
      if (!started.ok || !response.threadId) throw new Error(response.error);

      threadId.current = response.threadId;
      const stream = await fetch(
        `/api/chat/${response.threadId}/events?after=${response.cursor ?? 0}`
      );
      if (!stream.ok || !stream.body) {
        throw new Error("Could not open the response stream.");
      }

      const reader = stream.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let finished = false;

      while (!finished) {
        const { done, value } = await reader.read();
        buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });
        const boundary = buffer.lastIndexOf("\n\n");
        if (boundary >= 0) {
          const events = readEvents(buffer.slice(0, boundary));
          buffer = buffer.slice(boundary + 2);
          for (const event of events) {
            if (event.kind === "model.text_delta") {
              updateAssistant(
                assistantId,
                (current) => current + (event.payload.text ?? "")
              );
            }
            if (event.kind === "turn.completed") {
              updateAssistant(assistantId, () => event.payload.response ?? "");
              finished = true;
            }
            if (event.kind === "turn.failed" || event.kind === "turn.cancelled") {
              throw new Error(
                event.payload.error ?? "The response did not complete."
              );
            }
          }
        }
        if (done) finished = true;
      }
    } catch (cause) {
      const message =
        cause instanceof Error && cause.message
          ? cause.message
          : "Unable to reach Heathcliff.";
      setError(message);
      updateAssistant(
        assistantId,
        (current) => current || "I could not complete that request."
      );
    } finally {
      setIsStreaming(false);
    }
  }

  return (
    <AppShell hasConversation={messages.length > 0} onNewChat={startNewChat}>
      <main className="flex min-h-[calc(100svh-3rem)] flex-1 flex-col">
        {messages.length ? (
          <Conversation className="mx-auto w-full max-w-3xl flex-1 px-4">
            <ConversationContent>
              {messages.map((message) => (
                <Message from={message.from} key={message.id}>
                  <MessageContent>
                    {message.from === "assistant" ? (
                      <MessageResponse>{message.text || "…"}</MessageResponse>
                    ) : (
                      message.text
                    )}
                  </MessageContent>
                </Message>
              ))}
            </ConversationContent>
          </Conversation>
        ) : (
          <div className="flex flex-1 items-center justify-center px-4 pb-24">
            <h1 className="text-2xl font-medium tracking-tight">
              Ready when you are.
            </h1>
          </div>
        )}

        <div className="mx-auto w-full max-w-3xl px-4 pb-8">
          <PromptInput
            className="[&>[data-slot=input-group]]:rounded-3xl [&>[data-slot=input-group]]:bg-muted/70"
            onSubmit={(message) => submit(message.text)}
          >
            <PromptInputBody>
              <PromptInputTextarea placeholder="Message Heathcliff" />
            </PromptInputBody>
            <PromptInputFooter>
              <PromptInputTools>
                <PromptInputButton tooltip="Attachments are coming soon">
                  <PlusIcon />
                </PromptInputButton>
              </PromptInputTools>
              <PromptInputSubmit disabled={isStreaming} />
            </PromptInputFooter>
          </PromptInput>
          {error && (
            <p aria-live="polite" className="px-3 pt-2 text-sm text-destructive">
              {error}
            </p>
          )}
        </div>
      </main>
    </AppShell>
  );
}
