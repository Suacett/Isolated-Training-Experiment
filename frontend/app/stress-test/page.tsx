"use client";

import PortfolioComparison from "../components/PortfolioStressTest";
import { useRouter } from "next/navigation";

export default function LaboratoryPage() {
    const router = useRouter();

    return (
        <PortfolioComparison onBack={() => router.push("/")} />
    );
}
