import { AppShell } from "@/components/app-shell";

export default function MemoryPage() {
  return (
    <AppShell>
      <main className="mx-auto flex w-full max-w-3xl flex-1 flex-col px-4 pt-12">
        <h1 className="text-2xl font-medium tracking-tight">Memory</h1>
        <p className="mt-3 text-sm text-muted-foreground">No memories yet.</p>
      </main>
    </AppShell>
  );
}
