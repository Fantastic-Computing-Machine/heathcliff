"use client";

import Link from "next/link";
import type { ReactNode } from "react";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarInset,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarProvider,
  SidebarRail,
  SidebarTrigger,
} from "@/components/ui/sidebar";
import { TooltipProvider } from "@/components/ui/tooltip";
import {
  BrainIcon,
  LibraryIcon,
  MessageSquarePlusIcon,
  PanelLeftIcon,
  SettingsIcon,
} from "lucide-react";

const sidebarRow =
  "!bg-transparent hover:!bg-transparent active:!bg-transparent data-[active=true]:!bg-transparent";

type AppShellProps = {
  children: ReactNode;
  hasConversation?: boolean;
  onNewChat?: () => void;
};

export function AppShell({
  children,
  hasConversation = false,
  onNewChat,
}: AppShellProps) {
  return (
    <TooltipProvider>
      <SidebarProvider>
        <Sidebar collapsible="icon">
          <SidebarHeader>
            <div className="flex h-10 items-center justify-between px-2 group-data-[collapsible=icon]:px-0">
              <span className="font-semibold group-data-[collapsible=icon]:hidden">
                Heathcliff
              </span>
              <SidebarTrigger aria-label="Toggle sidebar">
                <PanelLeftIcon />
              </SidebarTrigger>
            </div>
            <SidebarMenu>
              <SidebarMenuItem>
                <SidebarMenuButton asChild className={sidebarRow} tooltip="New chat">
                  <Link href="/chat" onClick={onNewChat}>
                    <MessageSquarePlusIcon />
                    <span>New chat</span>
                  </Link>
                </SidebarMenuButton>
              </SidebarMenuItem>
            </SidebarMenu>
          </SidebarHeader>

          <SidebarContent>
            <SidebarGroup>
              <SidebarGroupLabel>Chats</SidebarGroupLabel>
              <SidebarGroupContent>
                <SidebarMenu>
                  <SidebarMenuItem>
                    <SidebarMenuButton className={sidebarRow} tooltip="Chats">
                      <LibraryIcon />
                      <span>{hasConversation ? "Current conversation" : "No conversations yet"}</span>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                </SidebarMenu>
              </SidebarGroupContent>
            </SidebarGroup>
          </SidebarContent>

          <SidebarFooter>
            <SidebarMenu>
              <SidebarMenuItem>
                <SidebarMenuButton asChild className={sidebarRow} tooltip="Memory">
                  <Link href="/memory">
                    <BrainIcon />
                    <span>Memory</span>
                  </Link>
                </SidebarMenuButton>
              </SidebarMenuItem>
              <SidebarMenuItem>
                <SidebarMenuButton asChild className={sidebarRow} tooltip="Settings">
                  <Link href="/settings">
                    <SettingsIcon />
                    <span>Settings</span>
                  </Link>
                </SidebarMenuButton>
              </SidebarMenuItem>
            </SidebarMenu>
          </SidebarFooter>
          <SidebarRail />
        </Sidebar>

        <SidebarInset className="min-h-svh bg-background">
          <header className="flex h-12 items-center px-4">
            <SidebarTrigger className="md:hidden" aria-label="Open sidebar" />
          </header>
          {children}
        </SidebarInset>
      </SidebarProvider>
    </TooltipProvider>
  );
}
