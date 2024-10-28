import Link from "next/link";
import { getPostData, getSortedPostsData } from "../../../lib/api";
import MarkdownRenderer from "../../MarkdownRenderer";
import { Scroll, Github } from "lucide-react";
import { Button } from "@/components/ui/button";

export async function generateStaticParams() {
  const posts = getSortedPostsData();
  return posts.map((post) => ({
    id: post.id,
  }));
}

export default function BlogPost({ params }: { params: { id: string } }) {
  const postData = getPostData(params.id);

  return (
    <main className="min-h-screen p-8">
      <div className="max-w-3xl mx-auto">
        <div className="flex items-center justify-between mb-8">
          <Link href="/blog" className="flex items-center">
            <Scroll className="w-10 h-10 mr-2 text-primary" strokeWidth={1.5} />
            <span className="text-2xl font-bold">docetl blog</span>
          </Link>
          <Button asChild variant="ghost" className="flex items-center">
            <a
              href="https://github.com/ucbepic/docetl"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center text-gray-600 hover:text-gray-900"
            >
              <Github className="w-6 h-6 mr-2" />
              <span>GitHub</span>
            </a>
          </Button>
        </div>
        <article>
          <h1 className="text-3xl font-bold mb-4">{postData.title}</h1>
          <div className="text-gray-500 mb-8">{postData.date}</div>
          <div className="mb-12">
            <MarkdownRenderer content={postData.content} />
          </div>
        </article>
        <div className="mt-12 space-y-4">
          <Link href="/blog" className="text-blue-500 hover:underline block">
            &larr; Back to blog
          </Link>
          <Link href="/" className="text-blue-500 hover:underline block">
            &larr; Back to home
          </Link>
        </div>
      </div>
    </main>
  );
}
