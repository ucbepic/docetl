import { NextRequest, NextResponse } from "next/server";
import fs from "fs/promises";

const CHUNK_SIZE = 500000; // Number of characters to read at a time

export async function GET(req: NextRequest) {
  const filePath = req.nextUrl.searchParams.get("path");
  const page = parseInt(req.nextUrl.searchParams.get("page") || "0", 10);

  if (!filePath) {
    return NextResponse.json({ error: "Invalid file path" }, { status: 400 });
  }

  try {
    const fileHandle = await fs.open(filePath, "r");
    const stats = await fileHandle.stat();
    const fileSize = stats.size;

    const start = page * CHUNK_SIZE;
    const buffer = Buffer.alloc(CHUNK_SIZE);
    const { bytesRead } = await fileHandle.read(buffer, 0, CHUNK_SIZE, start);

    await fileHandle.close();

    const content = buffer.toString("utf-8", 0, bytesRead);

    return NextResponse.json(
      {
        content,
        totalSize: fileSize,
        page,
        hasMore: start + bytesRead < fileSize,
      },
      { status: 200 }
    );
  } catch (error) {
    console.error("Error reading file:", error);
    return NextResponse.json({ error: "Failed to read file" }, { status: 500 });
  }
}
