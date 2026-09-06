import { isThreadId, runtimeUrl } from "@/lib/runtime";

export const dynamic = "force-dynamic";

export async function GET(
  request: Request,
  { params }: { params: Promise<{ threadId: string }> }
) {
  const { threadId } = await params;
  if (!isThreadId(threadId)) {
    return Response.json({ error: "Invalid conversation." }, { status: 400 });
  }

  const after = new URL(request.url).searchParams.get("after") ?? "0";
  try {
    const upstream = await fetch(
      `${runtimeUrl()}/v2/threads/${threadId}/events?after=${encodeURIComponent(after)}`,
      { cache: "no-store", headers: { Accept: "text/event-stream" } }
    );
    if (!upstream.ok || !upstream.body) {
      return Response.json({ error: "Could not open the response stream." }, { status: 502 });
    }
    return new Response(upstream.body, {
      headers: {
        "Cache-Control": "no-cache, no-transform",
        Connection: "keep-alive",
        "Content-Type": "text/event-stream; charset=utf-8",
      },
    });
  } catch (error) {
    console.error("Runtime V2 event stream failed", error);
    return Response.json({ error: "The Heathcliff runtime is unavailable." }, { status: 503 });
  }
}
