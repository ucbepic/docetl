declare module "react-force-graph-2d" {
  interface NodeObject {
    id: string | number;
    val?: number;
    [key: string]: unknown;
  }

  interface LinkObject {
    source: string | number | NodeObject;
    target: string | number | NodeObject;
    distance?: number;
    [key: string]: unknown;
  }

  interface ForceGraphProps {
    graphData: {
      nodes: NodeObject[];
      links: LinkObject[];
    };
    nodeRelSize?: number;
    nodeVal?: (node: NodeObject) => number;
    linkLabel?: (link: LinkObject) => string;
    linkWidth?: number;
    linkColor?: string | ((link: LinkObject) => string);
    nodeLabel?: (node: NodeObject) => string;
    backgroundColor?: string;
    linkDirectionalParticles?: number;
    linkDirectionalParticleWidth?: number;
    d3VelocityDecay?: number;
    d3Force?: (force: unknown) => void;
    [key: string]: unknown;
  }

  const ForceGraph2D: React.FC<ForceGraphProps>;
  export default ForceGraph2D;
}
