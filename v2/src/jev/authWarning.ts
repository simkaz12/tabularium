import { AuthenticationError, PermissionDeniedError } from "@typesafe-ai/sdk";

import type { JevClient } from "../domain.ts";

export const AUTH_WARNING_INTERVAL_MS = 60 * 60 * 1000;

interface AuthWarningOptions {
  now?: () => number;
  warn?: (line: string) => void;
  intervalMs?: number;
}

export function isAuthError(error: unknown): boolean {
  if (error instanceof AuthenticationError || error instanceof PermissionDeniedError) {
    return true;
  }

  const status = typeof error === "object" && error !== null ? (error as { status?: unknown }).status : undefined;

  return status === 401 || status === 403;
}

export function withAuthWarning(client: JevClient, options: AuthWarningOptions = {}): JevClient {
  const now = options.now ?? Date.now;
  const warn = options.warn ?? ((line: string) => console.warn(line));
  const intervalMs = options.intervalMs ?? AUTH_WARNING_INTERVAL_MS;
  let lastWarnedAt: number | null = null;

  return {
    async systemOne(request, requestOptions) {
      try {
        return await client.systemOne(request, requestOptions);
      } catch (error) {
        if (isAuthError(error) && (lastWarnedAt === null || now() - lastWarnedAt >= intervalMs)) {
          lastWarnedAt = now();
          warn(
            JSON.stringify({
              event: "jev-auth-error",
              message: "TypeSafe rejected the API key (401 or 403): Jev checks are off until it is fixed. Run: tabumcp key set",
            }),
          );
        }

        throw error;
      }
    },
  };
}
