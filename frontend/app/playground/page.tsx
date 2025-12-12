"use client";

import { useRouter } from "next/navigation";
import ModelPlayground from "../components/ModelPlayground";

export default function PlaygroundPage() {
    const router = useRouter();

    return (
        <div className="flex flex-col min-h-screen">
            {/* We pass a no-op or actual redirect for compatibility, 
                 though we plan to remove the Back button from component */}
            <ModelPlayground onBack={() => router.push('/')} />
        </div>
    );
}
