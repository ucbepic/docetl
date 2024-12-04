import { NextRequest, NextResponse } from "next/server";
import { writeFile } from "fs/promises";
import path from "path";
import { mkdir } from "fs/promises";
import os from "os";

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const file = formData.get("file") as File;
    const namespace = formData.get("namespace") as string;
    if (!file) {
      return NextResponse.json({ error: "No file uploaded" }, { status: 400 });
    }

    // Convert the file to buffer
    const bytes = await file.arrayBuffer();
    const buffer = Buffer.from(bytes);

    // Create uploads directory in user's home directory if it doesn't exist
    const homeDir = process.env.DOCETL_HOME_DIR || os.homedir();
    const uploadDir = path.join(homeDir, ".docetl", namespace, "files");
    await mkdir(uploadDir, { recursive: true });

    // Create full file path
    const filePath = path.join(uploadDir, file.name);

    // Write the file
    await writeFile(filePath, buffer);

    // Return the absolute path
    return NextResponse.json({ path: filePath });
  } catch (error) {
    console.error("Error uploading file:", error);
    return NextResponse.json(
      { error: "Failed to upload file" },
      { status: 500 }
    );
  }
}
