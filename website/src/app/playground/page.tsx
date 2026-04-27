import { redirect } from "next/navigation";
import { v4 as uuidv4 } from "uuid";

export default function PlaygroundPage() {
  const workspaceId = uuidv4();
  redirect(`/playground/${workspaceId}`);
}
