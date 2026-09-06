import { isThreadId, runtimeUrl } from "@/lib/runtime";

export const dynamic = "force-dynamic";

type ChatRequest = { content?: unknown; threadId?: unknown };

export async function POST(request: Request) {
  const body = (await request.json()) as ChatRequest;
  const content = typeof body.content === "string" ? body.content.trim() : "";

  if (!content || content.length > 10_000) {
    return Response.json({ error: "Enter a message of up to 10,000 characters." }, { status: 400 });
  }

  let threadId = typeof body.threadId === "string" ? body.threadId : undefined;
  if (threadId && !isThreadId(threadId)) {
    return Response.json({ error: "Invalid conversation." }, { status: 400 });
  }

  try {
    if (!threadId) {
      const created = await fetch(`${runtimeUrl()}/v2/threads`, {
        method: "POST",
        cache: "no-store",
      });
      if (!created.ok) throw new Error("Could not create a conversation.");
      ({ thread_id: threadId } = (await created.json()) as { thread_id: string });
    }

    const admitted = await fetch(`${runtimeUrl()}/v2/threads/${threadId}/turns`, {
      method: "POST",
      cache: "no-store",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content, idempotency_key: crypto.randomUUID() }),
    });
    if (!admitted.ok) throw new Error("Could not start the response.");

    const { event_cursor: cursor } = (await admitted.json()) as {
      event_cursor: string;
    };
    return Response.json({ threadId, cursor: Number(cursor) || 0 });
  } catch (error) {
    console.error("Runtime V2 chat request failed", error);
    return Response.json(
      { error: "The Heathcliff runtime is unavailable. Check that it is running on port 8700." },
      { status: 503 }
    );
  }
}
