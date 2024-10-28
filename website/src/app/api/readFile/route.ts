import { NextRequest, NextResponse } from "next/server";
import fs from "fs/promises";
import axios from "axios";

export async function GET(req: NextRequest) {
  const filePath = req.nextUrl.searchParams.get("path");

  if (!filePath) {
    return NextResponse.json({ error: "Invalid file path" }, { status: 400 });
  }

  try {
    let fileContent;
    if (filePath.startsWith("http")) {
      const response = await axios.get(filePath);
      fileContent = response.data;
    } else {
      fileContent = await fs.readFile(filePath, "utf-8");
    }
    return new NextResponse(fileContent, { status: 200 });
  } catch (error) {
    console.error("Error reading file:", error);
    return NextResponse.json({ error: "Failed to read file" }, { status: 500 });
  }
}
