import { NextResponse } from "next/server";
import fs from "fs";
import { getNamespaceDir } from "@/app/api/utils";
import os from "os";

export async function POST(request: Request) {
  try {
    const { namespace } = await request.json();
    const homeDir = process.env.DOCETL_HOME_DIR || os.homedir();

    if (!namespace) {
      return NextResponse.json(
        { error: "Namespace parameter is required" },
        { status: 400 }
      );
    }

    const namespaceDir = getNamespaceDir(homeDir, namespace);
    const exists = fs.existsSync(namespaceDir);

    if (!exists) {
      fs.mkdirSync(namespaceDir, { recursive: true });
    }

    return NextResponse.json({ exists });
  } catch (error) {
    console.error("Error checking/creating namespace:", error);
    return NextResponse.json(
      { error: "Failed to check/create namespace" },
      { status: 500 }
    );
  }
}
