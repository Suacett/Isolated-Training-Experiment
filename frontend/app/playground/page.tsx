"use client";

import { useRouter } from "next/navigation";
import ModelPlayground from "../components/ModelPlayground";

export default function PlaygroundPage() {
    const router = useRouter();

    return (
        <div className="flex flex-col min-h-screen">
            {/* Passing onBack to navigate back to dashboard */}
            <ModelPlayground onBack={() => router.push('/')} />
        </div>
    );
}
