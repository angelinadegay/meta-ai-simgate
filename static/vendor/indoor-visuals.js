import * as THREE from "three";

const COLORS = {tile:'#d6e0db',wood:'#b38e68',rug:'#36696e',wet:'#b7dbe0',wall:'#e4e8e3',counter:'#899792',base:'#aeb9b3'};

function floorMaterial(surface, size) {
  const canvas=document.createElement('canvas');canvas.width=canvas.height=256;
  const context=canvas.getContext('2d');context.fillStyle=COLORS[surface]||COLORS.wood;context.fillRect(0,0,256,256);
  if(surface==='tile') {
    context.strokeStyle='#a5b6ae';context.lineWidth=3;context.strokeRect(0,0,256,256);
  } else if(surface==='wood') {
    for(let index=0;index<24;index++){context.strokeStyle=index%2?'#9c7956':'#c3a17a';context.lineWidth=.8;
      context.beginPath();context.moveTo(index*11,0);context.lineTo(index*11+Math.sin(index)*4,256);context.stroke();}
    context.strokeStyle='#876a4d';context.strokeRect(0,0,128,256);
  } else if(surface==='rug') {
    for(let index=0;index<128;index++){context.fillStyle=index%2?'#467c7d':'#2f5c63';context.fillRect(index*2,0,1,256);}
  }
  const texture=new THREE.CanvasTexture(canvas);texture.colorSpace=THREE.SRGBColorSpace;
  texture.wrapS=texture.wrapT=THREE.RepeatWrapping;texture.repeat.set(size[0]/.5,size[1]/.5);
  const material=new THREE.MeshStandardMaterial({map:texture,color:0xffffff,roughness:surface==='wet'?.18:.88,
    metalness:0,transparent:surface==='wet',opacity:surface==='wet'?.8:1});
  material.userData.qualityMap=texture;material.userData.qualityHighColor=new THREE.Color(0xffffff);
  material.userData.qualityBaseColor=new THREE.Color(COLORS[surface]||COLORS.wood);
  return material;
}

export function decorateIndoor(sceneData, meshes) {
  const materials=new Map(sceneData.indoor.geoms.map(geom=>[geom.name,geom]));
  for(const geom of sceneData.geoms) {
    const mesh=meshes[geom.id],record=materials.get(geom.name);if(!mesh||!record)continue;
    mesh.material.dispose();
    if(record.surface==='wall') {
      const near=geom.name==='indoor_wall_south'||geom.name==='indoor_wall_east';
      mesh.material=new THREE.MeshStandardMaterial({color:COLORS.wall,roughness:.96,
        transparent:near,opacity:near?.32:1,depthWrite:!near});
    } else mesh.material=floorMaterial(record.surface,record.size_m);
    mesh.receiveShadow=true;mesh.castShadow=record.surface==='wall'||record.surface==='counter';
    if(geom.name==='indoor_kitchen_counter') {
      const basin=new THREE.Mesh(new THREE.BoxGeometry(.64,.42,.025),
        new THREE.MeshStandardMaterial({color:0x51656c,roughness:.3,metalness:.6}));
      basin.name='indoor_sink_visual';basin.position.set(-.1,0,.465);mesh.add(basin);
      const tap=new THREE.Mesh(new THREE.TorusGeometry(.09,.012,8,16,Math.PI),
        new THREE.MeshStandardMaterial({color:0xbac7c7,metalness:.7,roughness:.2}));
      tap.position.set(-.1,-.22,.55);tap.rotation.x=Math.PI/2;mesh.add(tap);
    }
  }
}

export function apartmentThumbnail(layout) {
  const canvas=document.createElement('canvas');canvas.width=420;canvas.height=300;
  const context=canvas.getContext('2d');context.fillStyle='#e7efeb';context.fillRect(0,0,420,300);
  if(!layout)return canvas.toDataURL();
  const scale=26,offsetX=210,offsetY=150;
  for(const geom of layout.geoms) {
    if(geom.surface==='base')continue;
    context.fillStyle=COLORS[geom.surface]||COLORS.wood;
    context.fillRect(offsetX+(geom.position[0]-geom.size_m[0]/2)*scale,
      offsetY-(geom.position[1]+geom.size_m[1]/2)*scale,geom.size_m[0]*scale,geom.size_m[1]*scale);
    if(geom.surface==='wall'){context.strokeStyle='#6c7d74';context.lineWidth=1;
      context.strokeRect(offsetX+(geom.position[0]-geom.size_m[0]/2)*scale,
        offsetY-(geom.position[1]+geom.size_m[1]/2)*scale,geom.size_m[0]*scale,geom.size_m[1]*scale);}
  }
  return canvas.toDataURL();
}