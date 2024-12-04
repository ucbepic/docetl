import { NextRequest, NextResponse } from "next/server";
import { writeFile } from "fs/promises";
import path from "path";
import { mkdir } from "fs/promises";
import os from "os";

interface SavedFile {
  name: string;
  path: string;
}

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const files = formData.getAll("files") as File[];
    const namespace = formData.get("namespace") as string;

    if (!files || files.length === 0) {
      return NextResponse.json({ error: "No files provided" }, { status: 400 });
    }

    const homeDir = process.env.DOCETL_HOME_DIR || os.homedir();

    // Create uploads directory in user's home directory if it doesn't exist
    const uploadsDir = path.join(homeDir, ".docetl", namespace, "documents");
    await mkdir(uploadsDir, { recursive: true });

    const savedFiles = await Promise.all(
      files.map(async (file) => {
        const bytes = await file.arrayBuffer();
        const buffer = Buffer.from(bytes);

        // Create a safe filename
        const fileName = file.name.replace(/[^a-zA-Z0-9.-]/g, "_");
        const filePath = path.join(uploadsDir, fileName);

        // Save the file
        await writeFile(filePath, buffer);

        return {
          name: file.name,
          path: filePath,
        } as SavedFile;
      })
    );

    return NextResponse.json({ files: savedFiles }, { status: 200 });
  } catch (error) {
    console.error("Error saving documents:", error);
    return NextResponse.json(
      { error: "Failed to save documents" },
      { status: 500 }
    );
  }
}
